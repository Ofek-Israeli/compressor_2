"""
Token embedder: map a list of tokens (or text files) to an embedding matrix using a configurable model.
Uses HuggingFace sentence-transformers. Token ordering is preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def embed_files(
    file_paths: list[str] | list[Path],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int | None = 128,
    **kwargs: object,
) -> np.ndarray:
    """
    Read text from each file, split by whitespace into tokens, and embed each token.

    Args:
        file_paths: Paths to text files. Order is preserved in the output (tokens from first file, then second, etc.).
        model_name: HuggingFace model name (e.g. "all-MiniLM-L6-v2"). Cached after first load.
        batch_size: Batch size for encode(); default 128 for better GPU utilization. Overridden by batch_size in kwargs.
        **kwargs: Passed through to SentenceTransformer.encode() (e.g. show_progress_bar).

    Returns:
        NumPy array of shape (n, embed_dim), dtype float32.
        If no texts to embed, returns shape (0, embed_dim).
    """
    if not file_paths:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embed_dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, embed_dim), dtype=np.float32)

    texts: list[str] = []
    for p in file_paths:
        path = Path(p)
        content = path.read_text(encoding="utf-8", errors="replace")
        texts.extend(content.split())

    if not texts:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embed_dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, embed_dim), dtype=np.float32)

    return embed_tokens(texts, model_name=model_name, batch_size=batch_size, **kwargs)


def embed_tokens(
    tokens: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int | None = None,
    **kwargs: object,
) -> np.ndarray:
    """
    Embed each token with the given sentence-transformers model.

    Args:
        tokens: List of token strings to embed. Order is preserved in the output.
        model_name: HuggingFace model name (e.g. "all-MiniLM-L6-v2"). Cached after first load.
        batch_size: Batch size for encode(); larger values can improve GPU utilization (e.g. 128 on Apple Silicon).
        **kwargs: Passed through to SentenceTransformer.encode() (e.g. show_progress_bar).

    Returns:
        NumPy array of shape (len(tokens), embed_dim), dtype float32.
        If tokens is empty, returns shape (0, embed_dim) where embed_dim is the model's dimension.
    """
    if not tokens:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embed_dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, embed_dim), dtype=np.float32)

    from sentence_transformers import SentenceTransformer

    encode_kwargs = dict(kwargs)
    if batch_size is not None:
        encode_kwargs["batch_size"] = batch_size
    model = SentenceTransformer(model_name)
    # Each token is embedded as a single "sentence" to get token-level embeddings
    embeddings = model.encode(tokens, **encode_kwargs)
    return np.asarray(embeddings, dtype=np.float32)
