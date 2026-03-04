"""
compressor_2: Embedding → PCA → Spherical K-means pipeline for token clustering.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
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
    "generate_processor",
    "get_representatives",
]


def generate_processor(*args, **kwargs):
    """Lazy wrapper — avoids importing joblib/transformers at package load."""
    from .logit_processor_generator import generate_processor as _gen
    return _gen(*args, **kwargs)


def run_pipeline(
    tokens: list[str],
    *,
    embedding_model: str = "all-MiniLM-L6-v2",
    pca_d: int,
    kmeans_k: int | None = None,
    random_state: int | None = None,
    spherical: bool = True,
    k_min: int = 5,
    k_max: int = 200,
    n_seeds: int = 5,
    device: str | None = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA, Union[KMeans, MiniBatchKMeans]]:
    """Run the full pipeline: embed tokens -> PCA -> spherical k-means.

    Args:
        tokens: List of token strings to process.
        embedding_model: HuggingFace sentence-transformers model name.
        pca_d: Number of PCA components.
        kmeans_k: Number of clusters.  None triggers auto-selection.
        random_state: Random state for PCA and KMeans (reproducibility).
        spherical: L2-normalize before clustering (cosine k-means).
        k_min, k_max: Range for auto-k search (when kmeans_k is None).
        n_seeds: Seeds per candidate k for stability (when kmeans_k is None).
        device: K-means device — ``"auto"`` | ``"cuda:N"`` | ``"cpu"``.

    Returns:
        (embeddings, reduced, labels, pca_fit, kmeans_fit)
    """
    X = embed_tokens(tokens, model_name=embedding_model)
    Z, pca = reduce_pca(X, d=pca_d, random_state=random_state)
    labels, kmeans = cluster_kmeans(
        Z,
        k=kmeans_k,
        random_state=random_state,
        spherical=spherical,
        k_min=k_min,
        k_max=k_max,
        n_seeds=n_seeds,
        device=device,
    )
    return X, Z, labels, pca, kmeans
