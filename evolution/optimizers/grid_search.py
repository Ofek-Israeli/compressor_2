"""
Grid search optimizer — exhaustive evaluation over a discretized box.

Builds a 1D grid per dimension, enumerates all points as a Cartesian product,
and streams them in batches through ``ctx.evaluate_batch``.

WARNING: The number of points is exponential in d (n_points ** d).
Use only for very small d or very coarse step.

See docs/zero_order_opt_plan.md §4, §9.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import List, Tuple

from ..config import EvolutionConfig
from ..eval_context import BudgetExceeded, EvalContext
from . import OptimizerResult

LOG = logging.getLogger(__name__)


def _build_1d_grid(low: float, high: float, step: float) -> List[float]:
    """Build a 1D grid [low, low+step, ..., <= high] with float-safe rounding."""
    n_points = int(math.floor((high - low) / step)) + 1
    return [round(low + k * step, 10) for k in range(n_points)]


def run(
    ctx: EvalContext,
    x0: List[float],
    bounds: Tuple[float, float],
    cfg: EvolutionConfig,
) -> OptimizerResult:
    d = len(ctx.cluster_ids)
    vals = _build_1d_grid(cfg.grid_low, cfg.grid_high, cfg.grid_step)
    n_points = len(vals)
    total_combos = n_points ** d

    LOG.info(
        "grid_search: d=%d, grid=[%.4f, %.4f] step=%.4f, "
        "n_points=%d, total_combos=%d, max_evals=%d, truncation=%s",
        d, cfg.grid_low, cfg.grid_high, cfg.grid_step,
        n_points, total_combos, ctx.max_evals, cfg.grid_allow_truncation,
    )

    if total_combos > cfg.grid_max_combos:
        raise ValueError(
            f"Grid search: total_combos ({total_combos}) exceeds "
            f"GRID_MAX_COMBOS ({cfg.grid_max_combos}). "
            f"n_points={n_points} per dimension, d={d} dimensions. "
            f"Use a coarser GRID_STEP, or set GRID_ALLOW_TRUNCATION=y and limit max_evals."
        )

    if total_combos > ctx.max_evals and not cfg.grid_allow_truncation:
        raise ValueError(
            f"Grid search: total_combos ({total_combos}) exceeds "
            f"max_evals ({ctx.max_evals}) and GRID_ALLOW_TRUNCATION is off. "
            f"Set CONFIG_GRID_ALLOW_TRUNCATION=y to evaluate partial grid."
        )

    batch: List[List[float]] = []
    batch_size = cfg.grid_batch_size

    try:
        for point in itertools.product(vals, repeat=d):
            x = list(point)
            batch.append(x)
            if len(batch) >= batch_size:
                ctx.evaluate_batch(batch)
                batch = []
        if batch:
            ctx.evaluate_batch(batch)
    except BudgetExceeded:
        LOG.info("grid_search: budget exhausted at %d/%d evals", ctx.n_evals, ctx.max_evals)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
