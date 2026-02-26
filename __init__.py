"""
compressor_2: Embedding → PCA → K-means pipeline for token clustering.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .embedder import embed_files, embed_tokens
from .kmeans_clusterer import cluster_kmeans
from .pca_reducer import reduce_pca
from .representatives import get_representatives

__all__ = [
    "embed_files",
    "embed_tokens",
    "reduce_pca",
    "cluster_kmeans",
    "run_pipeline",
    "get_representatives",
]


def run_pipeline(
    tokens: list[str],
    *,
    embedding_model: str = "all-MiniLM-L6-v2",
    pca_d: int,
    kmeans_k: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA, KMeans]:
    """
    Run the full pipeline: embed tokens → PCA to d dimensions → k-means with k clusters.

    Args:
        tokens: List of token strings to process.
        embedding_model: HuggingFace sentence-transformers model name.
        pca_d: Number of PCA components.
        kmeans_k: Number of k-means clusters.
        random_state: Random state for PCA and KMeans (reproducibility).

    Returns:
        (embeddings, reduced, labels, pca_fit, kmeans_fit)
        - embeddings: (n, embed_dim) from embed_tokens
        - reduced: (n, pca_d) from PCA
        - labels: (n,) cluster indices from k-means
        - pca_fit: fitted PCA object (e.g. to transform new data)
        - kmeans_fit: fitted KMeans object (e.g. to predict new points)
    """
    X = embed_tokens(tokens, model_name=embedding_model)
    Z, pca = reduce_pca(X, d=pca_d, random_state=random_state)
    labels, kmeans = cluster_kmeans(Z, k=kmeans_k, random_state=random_state)
    return X, Z, labels, pca, kmeans
