"""
Bayesian optimizer via scikit-optimize (skopt) with ask/tell.

Requires CONFIG_EVAL_DETERMINISTIC=y (surrogates need noiseless evaluations).

See docs/zero_order_opt_plan.md §4, §9.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..config import EvolutionConfig
from ..eval_context import BudgetExceeded, EvalContext
from . import OptimizerResult

LOG = logging.getLogger(__name__)


def run(
    ctx: EvalContext,
    x0: List[float],
    bounds: Tuple[float, float],
    cfg: EvolutionConfig,
) -> OptimizerResult:
    try:
        from skopt import Optimizer
    except ImportError:
        raise ImportError(
            "skopt method requires 'scikit-optimize'. "
            "Install with: pip install scikit-optimize"
        )

    d = len(ctx.cluster_ids)
    lo, hi = bounds
    n_calls = ctx.max_evals
    n_random = min(cfg.skopt_n_random_starts, n_calls)

    LOG.info(
        "skopt: d=%d, bounds=[%.4f, %.4f], n_calls=%d, n_random_starts=%d",
        d, lo, hi, n_calls, n_random,
    )

    opt = Optimizer(
        dimensions=[(lo, hi)] * d,
        n_initial_points=n_random,
        random_state=cfg.optimizer_seed,
    )

    opt.tell([x0], [-ctx.evaluate(x0)])

    try:
        while ctx.n_evals < ctx.max_evals:
            suggested = opt.ask()
            x = [float(v) for v in suggested]
            f = ctx.evaluate(x)
            opt.tell([x], [-f])
    except BudgetExceeded:
        LOG.info("skopt: budget exhausted at %d/%d evals", ctx.n_evals, ctx.max_evals)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
