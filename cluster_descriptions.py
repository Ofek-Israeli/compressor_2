"""
Generate short textual descriptions of clusters by sending representative
original tokens (strings) to an LLM (e.g. gpt-4o).
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np


def _top_words(text_units: list[str], indices: np.ndarray, n: int = 10) -> list[str]:
    """Return the *n* most frequent tokens in the cluster."""
    counts: Counter[str] = Counter()
    for i in indices:
        counts[text_units[i]] += 1
    return [word for word, _ in counts.most_common(n)]


def describe_clusters(
    labels: np.ndarray,
    reduced: np.ndarray,
    text_units: list[str],
    *,
    model: str = "gpt-4o",
    max_texts_per_cluster: int = 20,
    max_chars_per_cluster: int = 8000,
    top_n_example: int = 10,
) -> dict[int, dict]:
    """
    For each cluster, select representative text units (closest to centroid in reduced space),
    send them to the OpenAI Chat API, and return a description + example per cluster.

    Returns:
        Dict mapping cluster_id (int) to
        ``{"description": str, "example": list[str]}``.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it to use --descriptions-out with kmeans."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    unique_labels = np.unique(labels)
    result: dict[int, dict] = {}

    for c in unique_labels:
        indices = np.where(labels == c)[0]
        if len(indices) == 0:
            result[int(c)] = {"description": "(empty cluster)", "example": []}
            continue

        example = _top_words(text_units, indices, n=top_n_example)

        Z_c = reduced[indices]
        centroid_c = Z_c.mean(axis=0)
        dists = np.linalg.norm(Z_c - centroid_c, axis=1)
        top_local = np.argsort(dists)[:max_texts_per_cluster]
        global_indices = indices[top_local]
        chosen = [text_units[i] for i in global_indices]

        # Trim to max_chars_per_cluster
        buf: list[str] = []
        total = 0
        sep = "\n\n"
        for s in chosen:
            add_len = len(s) + (len(sep) if buf else 0)
            if total + add_len > max_chars_per_cluster and buf:
                break
            buf.append(s)
            total += add_len

        texts_block = sep.join(buf)
        words = texts_block.split()
        if len(words) <= 15:
            result[int(c)] = {
                "description": " ".join(words) or "(no description)",
                "example": example,
            }
            continue

        system = (
            "You are a helpful assistant. Given a list of text excerpts that belong to the same cluster, "
            "Provide a one-sentence description of the cluster. Output only the description, with no preamble."
        )
        user = (
            "Here are representative texts from one cluster:\n\n"
            f"{texts_block}\n\n"
            "Summarize in one or two sentences what theme or topic these texts have in common."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            description = (response.choices[0].message.content or "").strip()
            result[int(c)] = {
                "description": description or "(no description)",
                "example": example,
            }
        except Exception as e:
            result[int(c)] = {
                "description": f"(error: {e})",
                "example": example,
            }

    return result
