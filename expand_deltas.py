"""
Expand deltas_examples.json to have one key per k-means cluster.

Ensures keys "0" through "n_clusters-1" exist (fills missing with 0.0).
Use when the deltas file has fewer keys than the k-means model so that
evolution sees the correct number of dimensions.

Usage:
  PYTHONPATH=. python -m compressor_2.expand_deltas \\
    --kmeans outputs/labels_kmeans.joblib \\
    --deltas outputs/deltas_examples.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def expand_deltas(kmeans_path: str, deltas_path: str, default_value: float = 0.0) -> int:
    """Ensure deltas has keys 0..n_clusters-1; fill missing with default_value. Returns n_clusters."""
    import joblib

    kmeans = joblib.load(kmeans_path)
    n_clusters = getattr(kmeans, "n_clusters", kmeans.cluster_centers_.shape[0])

    with open(deltas_path, encoding="utf-8") as f:
        deltas: dict[str, float] = json.load(f)

    added = 0
    for i in range(n_clusters):
        key = str(i)
        if key not in deltas:
            deltas[key] = default_value
            added += 1

    # Sort keys numerically for consistent output
    sorted_deltas = {str(i): deltas[str(i)] for i in range(n_clusters)}

    with open(deltas_path, "w", encoding="utf-8") as f:
        json.dump(sorted_deltas, f, indent=2, ensure_ascii=False)
        f.write("\n")

    LOG.info("Expanded %s to %d keys (added %d)", deltas_path, n_clusters, added)
    return n_clusters


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Expand deltas JSON to one key per k-means cluster (in-place).",
    )
    p.add_argument("--kmeans", required=True, type=Path, help="Path to labels_kmeans.joblib")
    p.add_argument("--deltas", required=True, type=Path, help="Path to deltas_examples.json")
    p.add_argument(
        "--default",
        type=float,
        default=0.0,
        help="Default delta for added keys (default: 0.0)",
    )
    args = p.parse_args()

    if not args.kmeans.exists():
        print(f"Error: k-means file not found: {args.kmeans}", file=sys.stderr)
        sys.exit(1)
    if not args.deltas.exists():
        print(f"Error: deltas file not found: {args.deltas}", file=sys.stderr)
        sys.exit(1)

    n = expand_deltas(str(args.kmeans), str(args.deltas), default_value=args.default)
    print(f"Deltas now have {n} keys (0..{n - 1})", file=sys.stderr)


if __name__ == "__main__":
    main()
