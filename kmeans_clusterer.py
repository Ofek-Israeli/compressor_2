"""
K-means clusterer: cluster reduced embeddings into k clusters. Returns labels and fitted KMeans object.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_kmeans(
    Z: np.ndarray,
    k: int,
    random_state: int | None = None,
    *,
    n_init: int | str = 10,
) -> tuple[np.ndarray, KMeans]:
    """
    Fit k-means with k clusters on Z and return cluster labels and the fitted KMeans object.

    Args:
        Z: Reduced embedding matrix of shape (n, d), e.g. output of reduce_pca().
        k: Number of clusters. Must satisfy 1 <= k <= n (n = Z.shape[0]).
        random_state: Random state for reproducibility (passed to sklearn KMeans).
        n_init: Number of k-means runs with different seeds; 10 (default) or 'auto'.

    Returns:
        (labels, kmeans) where labels is np.ndarray of shape (n,) dtype int (cluster indices 0..k-1),
        and kmeans is the fitted KMeans instance.
        Use kmeans.predict(new_Z) to assign new points to the nearest cluster.

    Note:
        If k > n, sklearn will raise; ensure k <= number of samples.
    """
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
    )
    kmeans.fit(Z)
    return np.asarray(kmeans.labels_, dtype=np.int_), kmeans
