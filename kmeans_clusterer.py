"""
Spherical k-means clusterer with optional automatic k selection.

L2-normalizes embeddings before clustering (cosine-distance k-means).
When k is not specified, sweeps k_min..k_max and selects the best k
using silhouette (cosine), Davies-Bouldin, and stability (Adjusted Rand
Index across multiple seeds).

Supports GPU-accelerated k-means via PyTorch when a CUDA device is
available (pass ``device="cuda:0"`` or let it auto-detect).  The fitted
model is always returned as a scikit-learn KMeans object so that
joblib serialization and ``.predict()`` work unchanged.
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

LOG = logging.getLogger(__name__)

_MINIBATCH_THRESHOLD = 10_000


def _l2_normalize(Z: np.ndarray) -> np.ndarray:
    return normalize(Z, norm="l2", axis=1)


# ---------------------------------------------------------------------------
# GPU (PyTorch) k-means
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _resolve_device(device: Optional[str]) -> Optional[str]:
    """Return a concrete device string or None (= CPU/sklearn path)."""
    if device is None:
        return "cuda:0" if _gpu_available() else None
    if device == "cpu":
        return None
    if device.startswith("cuda") and not _gpu_available():
        LOG.warning("CUDA requested (%s) but torch.cuda.is_available() is False; using CPU.", device)
        return None
    return device


def _torch_kmeanspp(
    X: "torch.Tensor",
    k: int,
    rng: np.random.RandomState,
    x_sq: "torch.Tensor",
) -> "torch.Tensor":
    """K-means++ initialisation on GPU. Returns (k, d) centroid tensor."""
    import torch  # noqa: F811

    n = X.shape[0]
    idx = rng.randint(0, n)
    centroids = [X[idx]]

    for _ in range(1, k):
        C = torch.stack(centroids)
        c_sq = (C * C).sum(dim=1)
        dists = x_sq.unsqueeze(1) + c_sq.unsqueeze(0) - 2.0 * (X @ C.T)
        min_dists = dists.min(dim=1).values.clamp(min=0.0)

        probs = min_dists.cpu().numpy().astype(np.float64)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(n, dtype=np.float64) / n
        idx = rng.choice(n, p=probs)
        centroids.append(X[idx])

    return torch.stack(centroids)


def _torch_kmeans_fit(
    Z: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Run k-means on *device* using PyTorch.

    Returns ``(labels, centroids, inertia, n_iter)`` as numpy arrays/scalars.
    """
    import torch

    n, d = Z.shape
    X = torch.from_numpy(Z).to(device)
    x_sq = (X * X).sum(dim=1)

    best_labels: Optional[np.ndarray] = None
    best_centroids: Optional[np.ndarray] = None
    best_inertia = float("inf")
    best_n_iter = 0

    rng = np.random.RandomState(random_state)

    for _ in range(n_init):
        seed = int(rng.randint(0, 2**31))
        centroids = _torch_kmeanspp(X, k, np.random.RandomState(seed), x_sq)

        n_iter = 0
        for iteration in range(max_iter):
            c_sq = (centroids * centroids).sum(dim=1)
            dists = x_sq.unsqueeze(1) + c_sq.unsqueeze(0) - 2.0 * (X @ centroids.T)
            labels = dists.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, device=X.device)
            new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), X)
            counts.scatter_add_(0, labels, torch.ones(n, device=X.device))

            empty = counts == 0
            counts[empty] = 1.0
            new_centroids /= counts.unsqueeze(1)
            if empty.any():
                new_centroids[empty] = centroids[empty]

            shift = ((new_centroids - centroids) ** 2).sum().item()
            centroids = new_centroids
            n_iter = iteration + 1
            if shift < tol:
                break

        c_sq = (centroids * centroids).sum(dim=1)
        dists = x_sq.unsqueeze(1) + c_sq.unsqueeze(0) - 2.0 * (X @ centroids.T)
        labels = dists.argmin(dim=1)
        inertia = dists.gather(1, labels.unsqueeze(1)).sum().item()

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.cpu().numpy()
            best_centroids = centroids.cpu().numpy()
            best_n_iter = n_iter

    assert best_labels is not None
    return best_labels, best_centroids, best_inertia, best_n_iter  # type: ignore[return-value]


def _wrap_sklearn(
    k: int,
    labels: np.ndarray,
    centroids: np.ndarray,
    inertia: float,
    n_iter: int,
    d: int,
    random_state: Optional[int],
    n_init: int,
) -> KMeans:
    """Wrap GPU-computed results into a fitted sklearn KMeans for
    serialisation (joblib) and ``.predict()`` compatibility."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    model.cluster_centers_ = centroids
    model.labels_ = labels
    model.inertia_ = inertia
    model.n_iter_ = n_iter
    model.n_features_in_ = d
    model._n_threads = 1  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Unified fit helper (GPU or CPU)
# ---------------------------------------------------------------------------

def _make_kmeans(
    k: int,
    n: int,
    random_state: Optional[int],
    n_init: Union[int, str],
) -> Union[KMeans, MiniBatchKMeans]:
    if n > _MINIBATCH_THRESHOLD:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=int(n_init) if isinstance(n_init, str) and n_init != "auto" else n_init,
            batch_size=min(4096, n),
        )
    return KMeans(n_clusters=k, random_state=random_state, n_init=n_init)


def _fit_kmeans(
    Z: np.ndarray,
    k: int,
    random_state: Optional[int],
    n_init: Union[int, str],
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Union[KMeans, MiniBatchKMeans]]:
    """Fit k-means on *Z*, returning ``(labels, fitted_model)``.

    When *device* points to a CUDA GPU the heavy lifting runs in PyTorch;
    the returned model is a plain sklearn ``KMeans`` so that downstream
    code (joblib dump, ``.predict()``) is unaffected.
    """
    n_init_int = int(n_init) if isinstance(n_init, str) and n_init != "auto" else n_init
    if isinstance(n_init_int, str):
        n_init_int = 10

    if device is not None:
        labels, centroids, inertia, n_iter = _torch_kmeans_fit(
            Z, k,
            n_init=n_init_int,
            random_state=random_state,
            device=device,
        )
        model = _wrap_sklearn(k, labels, centroids, inertia, n_iter,
                              Z.shape[1], random_state, n_init_int)
        return labels, model

    model = _make_kmeans(k, Z.shape[0], random_state, n_init)
    model.fit(Z)
    return np.asarray(model.labels_, dtype=np.int_), model


# ---------------------------------------------------------------------------
# Auto-k selection
# ---------------------------------------------------------------------------

def auto_select_k(
    Z_norm: np.ndarray,
    k_min: int = 5,
    k_max: int = 200,
    n_seeds: int = 5,
    random_state: Optional[int] = None,
    n_init: Union[int, str] = 10,
    stability_threshold: float = 0.8,
    silhouette_tolerance: float = 0.02,
    device: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Select k via internal metrics and stability analysis.

    A) For each candidate k, compute silhouette (cosine) and Davies-Bouldin.
    B) For each candidate k, run clustering n_seeds times and compute mean
       pairwise Adjusted Rand Index as a stability measure.
    Pick the smallest k with ARI >= stability_threshold and silhouette
    within silhouette_tolerance of the best observed silhouette.

    Returns (best_k, diagnostics) where diagnostics maps each candidate k
    to its scores.
    """
    n = Z_norm.shape[0]
    k_max = min(k_max, n - 1)
    if k_min > k_max:
        k_min = k_max

    rng = np.random.RandomState(random_state)
    candidates = list(range(k_min, k_max + 1))

    diagnostics: Dict[str, Any] = {"per_k": {}}
    best_sil = -1.0

    backend = "GPU (%s)" % device if device else "CPU"
    LOG.info(
        "auto_select_k [%s]: scanning k=%d..%d (%d candidates, n_seeds=%d)",
        backend, k_min, k_max, len(candidates), n_seeds,
    )

    for k in candidates:
        seeds = [int(rng.randint(0, 2**31)) for _ in range(n_seeds)]

        all_labels: List[np.ndarray] = []
        for seed in seeds:
            lbl, _ = _fit_kmeans(Z_norm, k, seed, n_init, device=device)
            all_labels.append(np.asarray(lbl, dtype=np.int_))

        primary_labels = all_labels[0]
        sil = float(silhouette_score(Z_norm, primary_labels, metric="cosine"))
        db = float(davies_bouldin_score(Z_norm, primary_labels))

        if n_seeds >= 2:
            ari_pairs = [
                adjusted_rand_score(a, b) for a, b in combinations(all_labels, 2)
            ]
            mean_ari = float(np.mean(ari_pairs))
        else:
            mean_ari = 1.0

        diagnostics["per_k"][k] = {
            "silhouette_cosine": round(sil, 5),
            "davies_bouldin": round(db, 5),
            "mean_ari": round(mean_ari, 5),
        }

        if sil > best_sil:
            best_sil = sil

        LOG.info(
            "  k=%d  sil=%.4f  db=%.4f  ari=%.4f", k, sil, db, mean_ari,
        )
        print(f"k={k} calculated (sil={sil:.4f}, db={db:.4f}, ari={mean_ari:.4f})", file=sys.stderr)

    sil_floor = best_sil - silhouette_tolerance
    best_k = candidates[-1]
    for k in candidates:
        info = diagnostics["per_k"][k]
        if info["mean_ari"] >= stability_threshold and info["silhouette_cosine"] >= sil_floor:
            best_k = k
            break

    diagnostics["best_k"] = best_k
    diagnostics["best_silhouette"] = best_sil
    diagnostics["silhouette_floor"] = round(sil_floor, 5)
    diagnostics["stability_threshold"] = stability_threshold

    LOG.info(
        "auto_select_k: selected k=%d (sil=%.4f, ari=%.4f)",
        best_k,
        diagnostics["per_k"][best_k]["silhouette_cosine"],
        diagnostics["per_k"][best_k]["mean_ari"],
    )
    return best_k, diagnostics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_kmeans(
    Z: np.ndarray,
    k: Optional[int] = None,
    random_state: Optional[int] = None,
    *,
    n_init: Union[int, str] = 10,
    spherical: bool = True,
    k_min: int = 5,
    k_max: int = 200,
    n_seeds: int = 5,
    device: Optional[str] = "auto",
) -> Tuple[np.ndarray, Union[KMeans, MiniBatchKMeans]]:
    """Fit (spherical) k-means on Z and return (labels, fitted_model).

    When *spherical* is True (default), Z is L2-normalized before clustering
    so that k-means effectively operates with cosine distance.

    When *k* is None, :func:`auto_select_k` sweeps k_min..k_max using
    silhouette (cosine), Davies-Bouldin and Adjusted Rand Index stability.
    The selected k and diagnostics are printed to stderr.

    *device* controls where k-means runs:

    * ``"auto"`` (default) — use CUDA if available, else CPU.
    * ``"cuda:0"``, ``"cuda:1"`` — specific GPU.
    * ``"cpu"`` — force CPU (sklearn).

    The returned model is always a sklearn ``KMeans``/``MiniBatchKMeans``
    so that joblib serialisation and ``.predict()`` work unchanged.

    Args:
        Z: Embedding matrix (n, d).
        k: Number of clusters.  None triggers auto-selection.
        random_state: Seed for reproducibility.
        n_init: KMeans restarts per fit.
        spherical: L2-normalize before clustering (recommended).
        k_min, k_max: Range for auto-k search.
        n_seeds: Seeds per candidate k for stability measurement.
        device: ``"auto"`` | ``"cuda:N"`` | ``"cpu"``.

    Returns:
        (labels, fitted_model).  labels is np.ndarray of shape (n,).
    """
    resolved = _resolve_device(device if device != "auto" else None)
    backend = f"GPU ({resolved})" if resolved else "CPU (sklearn)"
    print(f"K-means backend: {backend}", file=sys.stderr)
    Z_fit = _l2_normalize(Z) if spherical else Z

    if k is None:
        k, diag = auto_select_k(
            Z_fit,
            k_min=k_min,
            k_max=k_max,
            n_seeds=n_seeds,
            random_state=random_state,
            n_init=n_init,
            device=resolved,
        )
        print(f"Auto-selected k={k}", file=sys.stderr)
        best_info = diag["per_k"][k]
        print(
            f"  silhouette(cosine)={best_info['silhouette_cosine']:.4f}  "
            f"davies-bouldin={best_info['davies_bouldin']:.4f}  "
            f"stability(ARI)={best_info['mean_ari']:.4f}",
            file=sys.stderr,
        )

    labels, model = _fit_kmeans(Z_fit, k, random_state, n_init, device=resolved)
    return np.asarray(labels, dtype=np.int_), model
