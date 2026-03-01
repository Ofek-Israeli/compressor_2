"""
Hybrid optimizer: global (DE / CMA-ES / Optuna-TPE) -> local (TR-DFO).

Phase 1: run the global method with ``hybrid_global_evals`` budget.
Phase 2: run TR-DFO starting from phase 1's best, using remaining budget.

See docs/zero_order_opt_plan.md §4.
"""

from __future__ import annotations

import json
import logging
from typing import List, Tuple

from ..config import EvolutionConfig
from ..eval_context import EvalContext
from . import OptimizerResult, get_optimizer

LOG = logging.getLogger(__name__)

_GLOBAL_METHODS = {"differential_evolution", "cmaes", "optuna_tpe"}


def run(
    ctx: EvalContext,
    x0: List[float],
    bounds: Tuple[float, float],
    cfg: EvolutionConfig,
) -> OptimizerResult:
    global_method = cfg.hybrid_global_method
    if global_method not in _GLOBAL_METHODS:
        raise ValueError(
            f"hybrid_global_method must be one of {sorted(_GLOBAL_METHODS)}, "
            f"got {global_method!r}"
        )

    total_budget = ctx.max_evals
    global_budget = cfg.hybrid_global_evals or (total_budget // 2)
    global_budget = min(global_budget, total_budget)

    LOG.info(
        "hybrid: global=%s (%d evals), local=tr_dfo (remaining), total=%d",
        global_method, global_budget, total_budget,
    )

    orig_max = ctx.max_evals
    ctx.max_evals = min(global_budget, orig_max)

    global_fn = get_optimizer(global_method)
    global_result = global_fn(ctx, x0, bounds, cfg)

    local_start = ctx.best_x or global_result.best_x
    remaining = orig_max - ctx.n_evals
    ctx.max_evals = orig_max

    if remaining <= 0:
        LOG.info("hybrid: no budget remaining for local phase")
        return OptimizerResult(
            best_x=ctx.best_x or x0,
            best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
            n_evals_used=ctx.n_evals,
            history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
        )

    # Record eval index where phase 2 starts (for zero_order_fitness.png vertical line)
    try:
        (ctx.out_dir / "hybrid_phase_switch.json").write_text(
            json.dumps({"eval_id": ctx.n_evals}),
        )
    except Exception:
        pass

    LOG.info("hybrid: starting local TR-DFO phase with %d remaining evals", remaining)
    local_fn = get_optimizer("tr_dfo")
    local_fn(ctx, local_start, bounds, cfg)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
