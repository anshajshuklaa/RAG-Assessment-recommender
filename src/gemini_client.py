"""
Shared Gemini client (google-genai SDK).

Single place for model names so the embedding model used to build the FAISS
index always matches the one used to embed queries.
"""

import hashlib
import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Embedding provider: "gemini" (API, free-tier quota), "local" (sentence-transformers, no key)
# or "hf" (Hugging Face Inference API, needs HF_TOKEN). The index must be built with the same one.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
_DEFAULT_MODELS = {"gemini": "gemini-embedding-001", "local": "BAAI/bge-base-en-v1.5", "hf": "BAAI/bge-base-en-v1.5"}
if EMBEDDING_PROVIDER not in _DEFAULT_MODELS:
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {EMBEDDING_PROVIDER!r}; use one of {sorted(_DEFAULT_MODELS)}")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or (
    os.getenv("GEMINI_EMBEDDING_MODEL", _DEFAULT_MODELS["gemini"]) if EMBEDDING_PROVIDER == "gemini"
    else _DEFAULT_MODELS[EMBEDDING_PROVIDER]
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
# BGE models expect this instruction on queries (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def index_path() -> str:
    """FAISS index file for the active embedding model, so indexes from different models never mix."""
    if EMBEDDING_PROVIDER == "gemini" and EMBEDDING_MODEL == "gemini-embedding-001":
        return "outputs/faiss_gemini_001.index"
    slug = EMBEDDING_MODEL.split("/")[-1].replace(".", "_").replace("-", "_")
    return f"outputs/faiss_{slug}.index"
GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Please set it in .env file")
    return genai.Client(api_key=api_key)


_CACHE_PATH = Path(os.getenv("EMBEDDING_CACHE_PATH", ".cache/query_embeddings.pkl"))
_cache: Optional[Dict[str, np.ndarray]] = None


def _cache_key(text: str, task_type: str) -> str:
    return hashlib.sha1(f"{EMBEDDING_MODEL}|{EMBEDDING_DIM}|{task_type}|{text}".encode()).hexdigest()


def _load_cache() -> Dict[str, np.ndarray]:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, "rb") as f:
                _cache = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            _cache = {}
    return _cache


def embed(texts: Union[str, List[str]], task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    """
    Embed text(s) and return L2-normalised float32 vectors of shape (n, EMBEDDING_DIM).

    Query embeddings are cached on disk, so a repeated query gets the identical vector
    (deterministic retrieval) without another API call.
    """
    if isinstance(texts, str):
        texts = [texts]
    use_cache = task_type == "RETRIEVAL_QUERY"
    cache = _load_cache() if use_cache else {}
    keys = [_cache_key(t, task_type) for t in texts]
    missing = [t for t, k in zip(texts, keys) if k not in cache]

    if missing:
        vectors = _embed_remote(missing, task_type)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"{EMBEDDING_MODEL} returned {vectors.shape[1]} dims, expected EMBEDDING_DIM={EMBEDDING_DIM}")
        if not use_cache:
            return vectors
        for text, vector in zip(missing, vectors):
            cache[_cache_key(text, task_type)] = vector
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
        except OSError:
            pass  # cache is an optimisation only

    return np.stack([cache[k] for k in keys])


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer  # optional dependency

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _hf_client():
    from huggingface_hub import InferenceClient  # optional dependency

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment (needed for EMBEDDING_PROVIDER=hf)")
    return InferenceClient(model=EMBEDDING_MODEL, token=token)


def _embed_remote(texts: List[str], task_type: str) -> np.ndarray:
    """Raw (unnormalised) embeddings from the configured provider."""
    if EMBEDDING_PROVIDER == "gemini":
        response = get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM),
        )
        return np.array([e.values for e in response.embeddings], dtype="float32")
    if task_type == "RETRIEVAL_QUERY" and "bge" in EMBEDDING_MODEL.lower():
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    if EMBEDDING_PROVIDER == "local":
        return np.asarray(_local_model().encode(texts), dtype="float32")
    return np.asarray(_hf_client().feature_extraction(texts), dtype="float32").reshape(len(texts), -1)


def generate(prompt: str, model: str = None) -> str:
    """Deterministic (temperature 0) text generation."""
    response = get_client().models.generate_content(
        model=model or GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text or ""


_LLM_CACHE_PATH = Path(os.getenv("LLM_CACHE_PATH", ".cache/llm_responses.pkl"))
_llm_cache: Optional[Dict[str, object]] = None


def _load_llm_cache() -> Dict[str, object]:
    global _llm_cache
    if _llm_cache is None:
        try:
            with open(_LLM_CACHE_PATH, "rb") as f:
                _llm_cache = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            _llm_cache = {}
    return _llm_cache


def generate_json(prompt: str, model: str = None, schema=None):
    """
    Temperature-0 generation constrained to JSON (optionally to a response schema); returns parsed JSON.

    Responses are cached on disk by (model, schema, prompt): a repeated request is answered
    instantly, costs no quota, and always returns the same output.
    """
    model = model or GENERATION_MODEL
    key = hashlib.sha1(f"{model}|{schema}|{prompt}".encode()).hexdigest()
    cache = _load_llm_cache()
    if key in cache:
        return cache[key]

    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    if schema is not None:
        config.response_schema = schema
    response = get_client().models.generate_content(model=model, contents=prompt, config=config)
    result = response.parsed if response.parsed is not None else json.loads(response.text or "")

    cache[key] = result
    try:
        _LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LLM_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except OSError:
        pass  # cache is an optimisation only
    return result
