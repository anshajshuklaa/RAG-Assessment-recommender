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

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768
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
        response = get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=missing,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM),
        )
        vectors = np.array([e.values for e in response.embeddings], dtype="float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
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


def generate(prompt: str, model: str = None) -> str:
    """Deterministic (temperature 0) text generation."""
    response = get_client().models.generate_content(
        model=model or GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text or ""


def generate_json(prompt: str, model: str = None, schema=None):
    """Temperature-0 generation constrained to JSON (optionally to a response schema); returns parsed JSON."""
    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    if schema is not None:
        config.response_schema = schema
    response = get_client().models.generate_content(
        model=model or GENERATION_MODEL,
        contents=prompt,
        config=config,
    )
    if response.parsed is not None:
        return response.parsed
    return json.loads(response.text or "")
