"""
Add an EOS/EOT cluster to compressor_2 pipeline artifacts.

Appends a new cluster (centroid computed from EOS/EOT token embeddings)
to the k-means model and adds corresponding entries in the description
and embeddings files so evolution/optimization can tune its delta.

Modifies in-place:
  - labels_kmeans.joblib: appends EOS/EOT centroid
  - cluster_descriptions.json: adds EOS/EOT description
  - embeddings.npy: appends EOS/EOT token embeddings

The new cluster ID = current n_clusters (appended at the end).

Usage:
  PYTHONPATH=/workspace python -m compressor_2.add_eos_eot_cluster \\
    --kmeans outputs/labels_kmeans.joblib \\
    --cluster-descriptions outputs/cluster_descriptions.json \\
    --embeddings outputs/embeddings.npy \\
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
    --embedding-model all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

EOS_EOT_DESCRIPTION = {
    "description": "End-of-sequence (EOS) and end-of-text (EOT) special tokens.",
    "example": ["<|endoftext|>", "<|eot_id|>", "<EOS>", "<EOT>"],
}


def _resolve_eos_eot_strings(tokenizer) -> list[str]:
    """Decode EOS and EOT token IDs to strings via the tokenizer."""
    strings: list[str] = []

    if tokenizer.eos_token_id is not None:
        s = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
        strings.append(s)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        s = tokenizer.decode([eot_id], skip_special_tokens=False)
        if s not in strings:
            strings.append(s)

    if not strings:
        raise ValueError("Could not find EOS or EOT tokens in the tokenizer.")
    return strings


def add_eos_eot_cluster(
    kmeans_path: str,
    cluster_descriptions_path: str,
    embeddings_path: str,
    tokenizer_name: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    pca_path: str | None = None,
) -> str:
    """Add an EOS/EOT cluster to the k-means model, descriptions, and embeddings.

    Returns the new cluster ID (string).
    """
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    # ---- load k-means model ----
    kmeans = joblib.load(kmeans_path)
    old_k = kmeans.n_clusters
    d = kmeans.cluster_centers_.shape[1]
    new_id = str(old_k)
    LOG.info("Existing k-means: %d clusters, %d-dim centroids → new cluster ID: %s", old_k, d, new_id)

    # ---- load current descriptions ----
    with open(cluster_descriptions_path, encoding="utf-8") as f:
        descriptions: dict[str, dict] = json.load(f)

    # ---- load embeddings ----
    X = np.load(embeddings_path)
    LOG.info("Existing embeddings shape: %s", X.shape)

    # ---- embed EOS/EOT tokens ----
    from transformers import AutoTokenizer

    from .embedder import embed_tokens

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_eot_strings = _resolve_eos_eot_strings(tokenizer)
    LOG.info("EOS/EOT token strings: %s", eos_eot_strings)

    X_eos = embed_tokens(eos_eot_strings, model_name=embedding_model)
    LOG.info("EOS/EOT raw embeddings shape: %s", X_eos.shape)

    embed_dim = X.shape[1]
    if X_eos.shape[1] != embed_dim:
        raise ValueError(
            f"EOS/EOT embeddings have {X_eos.shape[1]} dimensions but existing "
            f"embeddings.npy has {embed_dim}. Use the same --embedding-model as the "
            "pipeline (e.g. all-mpnet-base-v2 for 768-dim, all-MiniLM-L6-v2 for 384-dim)."
        )

    # ---- PCA-transform + L2-normalise → centroid ----
    if pca_path is not None:
        pca = joblib.load(pca_path)
        LOG.info("PCA loaded from %s (n_components=%s)", pca_path, pca.n_components_)
    else:
        from .pca_reducer import reduce_pca

        LOG.info("Re-fitting PCA(d=%d) from embeddings to project EOS/EOT …", d)
        _, pca = reduce_pca(X, d=d)

    Z_eos = pca.transform(X_eos)
    Z_eos = normalize(Z_eos, norm="l2", axis=1)
    new_centroid = normalize(
        Z_eos.mean(axis=0, keepdims=True), norm="l2", axis=1
    )
    LOG.info("EOS/EOT centroid computed (mean of %d vectors)", len(eos_eot_strings))

    # ---- extend k-means model ----
    new_centers = np.vstack([kmeans.cluster_centers_, new_centroid])
    new_kmeans = KMeans(n_clusters=old_k + 1, n_init=1)
    new_kmeans.cluster_centers_ = new_centers
    new_kmeans.n_features_in_ = d
    new_kmeans.labels_ = np.concatenate(
        [kmeans.labels_, np.array([old_k], dtype=kmeans.labels_.dtype)]
    )
    new_kmeans.inertia_ = getattr(kmeans, "inertia_", 0.0)
    new_kmeans.n_iter_ = getattr(kmeans, "n_iter_", 1)
    new_kmeans._n_threads = 1  # type: ignore[attr-defined]

    # ---- extend embeddings ----
    X_new = np.vstack([X, X_eos])

    # ---- extend descriptions ----
    descriptions[new_id] = EOS_EOT_DESCRIPTION

    # ---- save ----
    joblib.dump(new_kmeans, kmeans_path)
    LOG.info("Saved k-means (%d clusters) → %s", new_kmeans.n_clusters, kmeans_path)

    np.save(embeddings_path, X_new)
    LOG.info("Saved embeddings %s → %s", X_new.shape, embeddings_path)

    with open(cluster_descriptions_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
        f.write("\n")
    LOG.info("Saved descriptions → %s", cluster_descriptions_path)

    print(f"Added EOS/EOT cluster as ID {new_id}", file=sys.stderr)
    return new_id


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Add an EOS/EOT cluster to compressor_2 artifacts (in-place).",
    )
    p.add_argument(
        "--kmeans", required=True, type=Path,
        help="Path to labels_kmeans.joblib",
    )
    p.add_argument(
        "--cluster-descriptions", required=True, type=Path,
        help="Path to cluster_descriptions.json",
    )
    p.add_argument(
        "--embeddings", required=True, type=Path,
        help="Path to embeddings.npy",
    )
    p.add_argument(
        "--tokenizer", required=True,
        help="HuggingFace tokenizer name or path (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    p.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model; must match the pipeline (e.g. all-mpnet-base-v2 for 768-dim)",
    )
    p.add_argument(
        "--pca", default=None, type=Path,
        help="PCA joblib; if omitted, PCA is re-fit from --embeddings",
    )

    args = p.parse_args()

    add_eos_eot_cluster(
        kmeans_path=str(args.kmeans),
        cluster_descriptions_path=str(args.cluster_descriptions),
        embeddings_path=str(args.embeddings),
        tokenizer_name=args.tokenizer,
        embedding_model=args.embedding_model,
        pca_path=str(args.pca) if args.pca else None,
    )


if __name__ == "__main__":
    main()
