"""
PCA reducer: reduce embedding matrix to d dimensions. Returns transformed data and fitted PCA object.
Optimized for multi-core CPUs (e.g. Apple M4): uses randomized SVD when d is small for speed.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def reduce_pca(
    X: np.ndarray,
    d: int,
    random_state: int | None = None,
    *,
    svd_solver: str = "randomized",
    iterated_power: int = 4,
) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA with d components on X and return transformed data and the fitted PCA object.

    Args:
        X: Embedding matrix of shape (n, embed_dim), e.g. output of embed_tokens().
        d: Number of PCA components. Must satisfy 1 <= d <= X.shape[1].
        random_state: Random state for reproducibility (passed to sklearn PCA).
        svd_solver: 'randomized' (default) for fast approximate PCA on multi-core;
            use 'full' for exact SVD when needed.
        iterated_power: Power iterations for randomized solver (default 4); more = better accuracy, slower.

    Returns:
        (Z, pca) where Z is np.ndarray of shape (n, d) and pca is the fitted PCA instance.
        Use pca.transform(new_X) to transform new data with the same projection.

    Raises:
        ValueError: If d > X.shape[1] or d < 1.
    """
    n, embed_dim = X.shape
    if d < 1:
        raise ValueError(f"d must be >= 1, got d={d}")
    if d > embed_dim:
        raise ValueError(
            f"d ({d}) cannot exceed number of features (X.shape[1]={embed_dim}). "
            "Use d <= X.shape[1] or reduce embed_dim in the previous step."
        )

    pca = PCA(
        n_components=d,
        random_state=random_state,
        svd_solver=svd_solver,
        iterated_power=iterated_power,
    )
    Z = pca.fit_transform(X)
    return Z, pca
