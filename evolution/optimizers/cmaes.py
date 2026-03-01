"""
CMA-ES optimizer via the ``cma`` package.

Each generation: ask() -> ctx.evaluate_batch(candidates) -> tell().

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
        import cma
    except ImportError:
        raise ImportError(
            "cmaes method requires the 'cma' package. Install with: pip install cma"
        )

    d = len(ctx.cluster_ids)
    lo, hi = bounds
    sigma0 = (hi - lo) / 4.0

    LOG.info("cmaes: d=%d, sigma0=%.4f, budget=%d", d, sigma0, ctx.max_evals)

    opts = {
        "bounds": [[lo] * d, [hi] * d],
        "maxfevals": ctx.max_evals,
        "seed": cfg.optimizer_seed,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    try:
        while not es.stop():
            candidates = es.ask()
            xs = [list(c) for c in candidates]
            fitnesses = ctx.evaluate_batch(xs)
            es.tell(candidates, [-f for f in fitnesses])
    except BudgetExceeded:
        LOG.info("cmaes: budget exhausted at %d/%d evals", ctx.n_evals, ctx.max_evals)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
