"""
Trust-region derivative-free optimizer (BOBYQA / NEWUOA via pdfo or pybobyqa).

The backend calls the objective internally, so prefetch is not achievable;
each evaluation runs inline generate-processor.

See docs/zero_order_opt_plan.md §4, §9.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

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
    d = len(ctx.cluster_ids)
    lo, hi = bounds
    method = cfg.tr_dfo_method

    LOG.info("tr_dfo: d=%d, method=%s, bounds=[%.4f, %.4f], budget=%d", d, method, lo, hi, ctx.max_evals)

    def _obj(x_arr: np.ndarray) -> float:
        x = x_arr.tolist()
        f = ctx.evaluate(x)
        return -f

    lb = np.full(d, lo)
    ub = np.full(d, hi)
    x0_arr = np.array(x0)

    try:
        try:
            import pdfo
            LOG.info("Using pdfo backend (%s)", method)
            pdfo.pdfo(
                _obj, x0_arr,
                method=method,
                bounds=pdfo.Bounds(lb, ub),
                options={"maxfev": ctx.max_evals},
            )
        except ImportError:
            try:
                import pybobyqa
                LOG.info("Using pybobyqa backend")
                pybobyqa.solve(
                    _obj, x0_arr,
                    bounds=(lb, ub),
                    maxfun=ctx.max_evals,
                    seek_global_minimum=False,
                )
            except ImportError:
                raise ImportError(
                    "tr_dfo method requires 'pdfo' or 'pybobyqa'. "
                    "Install with: pip install pdfo  OR  pip install pybobyqa"
                )
    except BudgetExceeded:
        LOG.info("tr_dfo: budget exhausted at %d/%d evals", ctx.n_evals, ctx.max_evals)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
