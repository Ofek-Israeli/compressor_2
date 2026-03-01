"""
Differential evolution optimizer via scipy.

Uses scipy.optimize.differential_evolution with workers=1 and
updating='immediate'.  Each generation's population is evaluated via
``ctx.evaluate_batch`` for maximal prefetch.

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
    from scipy.optimize import differential_evolution as scipy_de

    d = len(ctx.cluster_ids)
    lo, hi = bounds
    scipy_bounds = [(lo, hi)] * d

    LOG.info("differential_evolution: d=%d, bounds=[%.4f, %.4f], budget=%d", d, lo, hi, ctx.max_evals)

    def _obj(x_arr: np.ndarray) -> float:
        x = x_arr.tolist()
        try:
            f = ctx.evaluate(x)
            return -f
        except BudgetExceeded:
            raise

    try:
        scipy_de(
            _obj,
            bounds=scipy_bounds,
            x0=np.array(x0),
            maxiter=ctx.max_evals,
            seed=cfg.optimizer_seed,
            workers=1,
            updating="immediate",
            tol=0,
            atol=0,
            polish=False,
        )
    except BudgetExceeded:
        LOG.info(
            "differential_evolution: budget exhausted at %d/%d evals",
            ctx.n_evals, ctx.max_evals,
        )
    except Exception:
        LOG.exception("differential_evolution: unexpected error")

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
