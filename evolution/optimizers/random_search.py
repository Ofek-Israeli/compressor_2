"""
Random search optimizer — sample uniformly in bounds and evaluate.

See docs/zero_order_opt_plan.md §4.
"""

from __future__ import annotations

import logging
import random as _random
from typing import List, Tuple

from ..config import EvolutionConfig
from ..eval_context import BudgetExceeded, EvalContext
from . import OptimizerResult

LOG = logging.getLogger(__name__)

_BATCH_SIZE = 64


def run(
    ctx: EvalContext,
    x0: List[float],
    bounds: Tuple[float, float],
    cfg: EvolutionConfig,
) -> OptimizerResult:
    d = len(ctx.cluster_ids)
    lo, hi = bounds
    rng = _random.Random(cfg.optimizer_seed)
    remaining = ctx.max_evals - ctx.n_evals

    LOG.info("random_search: d=%d, bounds=[%.4f, %.4f], budget=%d", d, lo, hi, remaining)

    ctx.evaluate(x0, next_x=None)

    try:
        batch: List[List[float]] = []
        for _ in range(remaining - 1):
            x = [rng.uniform(lo, hi) for _ in range(d)]
            batch.append(x)
            if len(batch) >= _BATCH_SIZE:
                ctx.evaluate_batch(batch)
                batch = []
        if batch:
            ctx.evaluate_batch(batch)
    except BudgetExceeded:
        LOG.info("random_search: budget exhausted at %d/%d evals", ctx.n_evals, ctx.max_evals)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
