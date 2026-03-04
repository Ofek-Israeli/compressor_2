"""
Build initial deltas from the k-means model and validate artifact consistency.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .config import EvolutionConfig

LOG = logging.getLogger(__name__)


def build_initial_deltas(cfg: EvolutionConfig) -> Tuple[Dict[str, float], List[str]]:
    """Derive initial deltas and cluster IDs from the k-means model.

    Returns (initial_deltas, cluster_ids) where every cluster gets
    cfg.initial_delta_value and cluster_ids is sorted numerically.
    """
    import joblib

    if not cfg.kmeans_path:
        raise ValueError("CONFIG_KMEANS_PATH is required to determine the number of clusters.")

    kmeans = joblib.load(cfg.kmeans_path)
    n_clusters = getattr(kmeans, "n_clusters", kmeans.cluster_centers_.shape[0])

    cluster_ids = [str(i) for i in range(n_clusters)]
    initial_deltas = {cid: cfg.initial_delta_value for cid in cluster_ids}

    LOG.info(
        "Built initial deltas from k-means (%s): %d clusters, initial_delta=%.4f",
        cfg.kmeans_path, n_clusters, cfg.initial_delta_value,
    )
    return initial_deltas, cluster_ids
