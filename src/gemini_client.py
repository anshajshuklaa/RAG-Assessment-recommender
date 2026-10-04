"""
Shared Gemini client (google-genai SDK).

Single place for model names so the embedding model used to build the FAISS
index always matches the one used to embed queries.
"""

import json
import os
from functools import lru_cache
from typing import List, Union

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


def embed(texts: Union[str, List[str]], task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    """Embed text(s) and return L2-normalised float32 vectors of shape (n, EMBEDDING_DIM)."""
    if isinstance(texts, str):
        texts = [texts]
    response = get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM),
    )
    vectors = np.array([e.values for e in response.embeddings], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors


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
