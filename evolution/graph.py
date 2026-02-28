"""
Matplotlib graph: composite score, correctness ratio, and mean length vs. iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

LOG = logging.getLogger(__name__)


def update_graph(
    history: List[Dict[str, Any]],
    output_dir: str,
    filename: str = "evolution_lengths.png",
) -> None:
    """(Re)draw the evolution graph and save as PNG.

    history: list of dicts with iteration, mean_token_length, correctness_ratio,
             shortness_score, composite_score.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    iters = [h["iteration"] for h in history]
    composite = [h["composite_score"] for h in history]
    correctness_ratio = [h["correctness_ratio"] for h in history]
    mean_length = [h["mean_token_length"] for h in history]

    best_composite = []
    best_correctness = []
    best_length = []
    best_so_far = -1.0
    best_idx = 0
    for i, c in enumerate(composite):
        if c > best_so_far:
            best_so_far = c
            best_idx = i
        best_composite.append(best_so_far)
        best_correctness.append(correctness_ratio[best_idx])
        best_length.append(mean_length[best_idx])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_top.plot(iters, best_composite, marker="o", linewidth=2, color="C0")
    ax_top.set_ylabel("Best composite score")
    ax_top.set_title("Best-so-far composite (0.4×shortness + 0.6×correctness)")
    ax_top.grid(True, alpha=0.3)
    if len(iters) > 1:
        ax_top.set_xticks(iters)

    ax_bottom.plot(iters, best_correctness, marker="s", linewidth=2, color="C1", label="Correctness ratio (best)")
    ax_bottom.set_ylabel("Correctness ratio")
    ax_bottom.set_ylim(-0.05, 1.05)
    ax_bottom.grid(True, alpha=0.3)

    ax_right = ax_bottom.twinx()
    ax_right.plot(iters, best_length, marker="^", linewidth=2, color="C2", label="Mean length (best)")
    ax_right.set_ylabel("Mean length (tokens)", color="C2")
    ax_right.tick_params(axis="y", labelcolor="C2")

    ax_bottom.set_xlabel("Evolution iteration")
    fig.legend(
        [ax_bottom.get_lines()[0], ax_right.get_lines()[0]],
        ["Correctness ratio (best)", "Mean length (best)"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )

    path = out / filename
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Graph updated: %s", path)
