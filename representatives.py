"""
Representatives retrieval: get the n closest points to each cluster centroid (in reduced space)
and return the corresponding original text units (tokens).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_text_units(text_path: str | Path, expected_len: int) -> list[str]:
    """
    Read the text file, split by whitespace into tokens, and return a list of N strings
    in the same order as the pipeline output.

    Args:
        text_path: Path to the original text file.
        expected_len: Expected number of units (must match labels / reduced rows).

    Returns:
        List of exactly expected_len strings.

    Raises:
        ValueError: If the number of units does not match expected_len.
    """
    path = Path(text_path)
    content = path.read_text(encoding="utf-8", errors="replace")
    units = content.split()
    if len(units) != expected_len:
        raise ValueError(
            f"Text file has {len(units)} units but labels/reduced have {expected_len}. "
            "The text source must have exactly the same number of units as the pipeline output, "
            "in the same order. Use the same source as when creating embeddings (whitespace-split tokens)."
        )
    return units


def load_text_units(text_path: str | Path, expected_len: int) -> list[str]:
    """
    Load text file into a list of token strings (whitespace-split).
    """
    return _load_text_units(text_path, expected_len)


def get_representatives(
    embeddings_path: str | Path,
    labels_path: str | Path,
    reduced_path: str | Path,
    text_path: str | Path,
    n: int,
    *,
    validate_embeddings: bool = True,
) -> dict[int, list[str]]:
    """
    Load the four artifacts and return, for each cluster label, the n closest representatives
    (by Euclidean distance to cluster centroid in reduced space) as original token strings.

    Args:
        embeddings_path: Path to embeddings .npy (used only for shape validation if validate_embeddings).
        labels_path: Path to labels .txt (one integer per line).
        reduced_path: Path to reduced .npy (used for centroid and distances).
        text_path: Path to the original text file (whitespace-split tokens, same order as pipeline output).
        n: Number of representatives to return per cluster (fewer if cluster is smaller).
        validate_embeddings: If True, load embeddings and check shape[0] == N; if False, skip.

    Returns:
        Dict mapping label_id (int) to list of at most n representative strings (closest to centroid).
    """
    path_reduced = Path(reduced_path)
    path_labels = Path(labels_path)

    reduced = np.load(path_reduced)
    if reduced.ndim != 2:
        raise ValueError(f"reduced.npy must be 2D, got shape {reduced.shape}")

    N = reduced.shape[0]
    labels_lines = path_labels.read_text().strip().splitlines()
    if len(labels_lines) != N:
        raise ValueError(
            f"labels.txt has {len(labels_lines)} lines but reduced has {N} rows."
        )
    labels = np.array([int(line.strip()) for line in labels_lines], dtype=np.int_)

    text_units = _load_text_units(text_path, N)

    if validate_embeddings:
        emb = np.load(Path(embeddings_path))
        if emb.shape[0] != N:
            raise ValueError(
                f"embeddings.npy has {emb.shape[0]} rows but expected {N}."
            )

    unique_labels = np.unique(labels)
    result: dict[int, list[str]] = {}

    for c in unique_labels:
        indices = np.where(labels == c)[0]
        Z_c = reduced[indices]
        centroid_c = Z_c.mean(axis=0)
        dists = np.linalg.norm(Z_c - centroid_c, axis=1)
        top_local = np.argsort(dists)[:n]
        global_indices = indices[top_local]
        result[int(c)] = [text_units[i] for i in global_indices]

    return result
