"""
Matplotlib graph: composite score, correctness ratio, and mean length vs. iteration.
Zero-order: fitness vs evaluation index (updated during run).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

ZERO_ORDER_FITNESS_FILENAME = "zero_order_fitness.png"
COORD_RD_GRAD_STATS_FILENAME = "zero_order_grad_stats.png"
VALIDATION_FITNESS_FILENAME = "validation_fitness.png"


def update_zero_order_fitness_graph(
    entries: List[Tuple[int, Optional[float], Optional[float]]],
    output_dir: str,
    method_name: str,
) -> None:
    """Plot fitness vs evaluation index and running best-so-far for zero-order methods.

    entries: list of (eval_id, f, best_f) per evaluation (f may be None on failure).
    output_dir: evolution output directory.
    method_name: e.g. 'differential_evolution' for title.
    Saves output_dir / zero_order_fitness.png. Optional: vertical line for hybrid phase 2.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not available; skipping zero_order_fitness.png")
        return

    if not entries:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eval_ids = [e[0] for e in entries]
    fs = [e[1] for e in entries]
    best_fs = [e[2] for e in entries]

    # Raw fitness: only successful evals
    eval_ids_ok = [e[0] for e in entries if e[1] is not None]
    fs_ok = [e[1] for e in entries if e[1] is not None]

    # Running best-so-far: use entry's best_f, or carry forward previous
    best_so_far = None
    best_line = []
    for b in best_fs:
        if b is not None:
            best_so_far = b
        best_line.append(best_so_far)

    fig, ax = plt.subplots(figsize=(8, 5))
    if eval_ids_ok and fs_ok:
        ax.scatter(eval_ids_ok, fs_ok, alpha=0.5, s=12, color="C0", label="Fitness")
    if best_so_far is not None:
        eval_ids_best = [eval_ids[i] for i in range(len(eval_ids)) if best_line[i] is not None]
        best_vals = [best_line[i] for i in range(len(eval_ids)) if best_line[i] is not None]
        ax.plot(eval_ids_best, best_vals, "-", linewidth=2, color="C2", label="Best so far")

    # Optional: hybrid phase switch vertical line
    phase_switch_path = out / "hybrid_phase_switch.json"
    if method_name == "hybrid" and phase_switch_path.exists():
        try:
            data = json.loads(phase_switch_path.read_text())
            pid = data.get("eval_id")
            if pid is not None and isinstance(pid, (int, float)):
                ax.axvline(x=int(pid), color="C3", linestyle="--", alpha=0.8, label="Phase 2 (TR-DFO)")
        except Exception:
            pass

    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{method_name.replace('_', ' ').title()}: Fitness vs evaluation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = out / ZERO_ORDER_FITNESS_FILENAME
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Zero-order graph updated: %s", path)


def update_coord_rd_grad_stats_graph(
    history_path: str,
    output_dir: str,
) -> None:
    """Plot derivative magnitude estimates for coordinate_then_random_direction.

    Reads zero_order_history.jsonl, extracts lines with mean_abs_grad_basis /
    mean_abs_grad_rand fields, and produces zero_order_grad_stats.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not available; skipping zero_order_grad_stats.png")
        return

    hist_path = Path(history_path)
    if not hist_path.exists():
        return

    indices: List[int] = []
    basis_vals: List[Optional[float]] = []
    rand_vals: List[Optional[float]] = []
    phases: List[str] = []
    idx = 0

    with open(hist_path, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "current_phase" not in rec:
                continue
            indices.append(rec.get("eval_id", idx))
            basis_vals.append(rec.get("mean_abs_grad_basis"))
            rand_vals.append(rec.get("mean_abs_grad_rand"))
            phases.append(rec.get("current_phase", ""))
            idx += 1

    if not indices:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    basis_x = [i for i, v in zip(indices, basis_vals) if v is not None]
    basis_y = [v for v in basis_vals if v is not None]
    rand_x = [i for i, v in zip(indices, rand_vals) if v is not None]
    rand_y = [v for v in rand_vals if v is not None]

    if basis_x:
        ax.plot(basis_x, basis_y, "-", linewidth=1.5, color="C0", label="mean |grad| basis")
    if rand_x:
        ax.plot(rand_x, rand_y, "-", linewidth=1.5, color="C1", label="mean |grad| random")

    # Vertical line at phase switch
    prev_phase = None
    for i, ph in zip(indices, phases):
        if prev_phase == "coordinate" and ph == "random_direction":
            ax.axvline(x=i, color="C3", linestyle="--", alpha=0.7, label="Phase switch")
            break
        prev_phase = ph

    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Mean |directional derivative|")
    ax.set_title("Coordinate-then-random-direction: derivative magnitude estimates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = out / COORD_RD_GRAD_STATS_FILENAME
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Grad stats graph updated: %s", path)


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


def plot_validation_fitness(
    rank_or_id: List[int],
    validation_fitness: List[float],
    output_path: Path,
    title: str = "Validation fitness (top-k from history)",
) -> None:
    """Plot validation fitness for top-k candidates (e.g. rank or eval_id on x-axis)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib not available; skipping validation_fitness.png")
        return

    if not rank_or_id or not validation_fitness or len(rank_or_id) != len(validation_fitness):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rank_or_id, validation_fitness, color="C0", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Rank (by training fitness)")
    ax.set_ylabel("Validation fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Validation fitness graph saved: %s", output_path)
