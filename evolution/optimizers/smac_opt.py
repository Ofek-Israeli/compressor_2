"""
SMAC (Sequential Model-based Algorithm Configuration) optimizer.

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
        from ConfigSpace import ConfigurationSpace, Float
        from smac import HyperparameterOptimizationFacade, Scenario
    except ImportError:
        raise ImportError(
            "smac method requires 'smac' and 'ConfigSpace' packages. "
            "Install with: pip install smac ConfigSpace"
        )

    d = len(ctx.cluster_ids)
    lo, hi = bounds
    n_trials = ctx.max_evals

    LOG.info("smac: d=%d, bounds=[%.4f, %.4f], n_trials=%d", d, lo, hi, n_trials)

    cs = ConfigurationSpace(seed=cfg.optimizer_seed)
    for i in range(d):
        cs.add(Float(f"x{i}", (lo, hi), default=x0[i]))

    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=n_trials,
        seed=cfg.optimizer_seed,
    )

    def _target(config, seed=0):
        x = [float(config[f"x{i}"]) for i in range(d)]
        f = ctx.evaluate(x)
        return -f

    facade = HyperparameterOptimizationFacade(scenario, _target, overwrite=True)

    try:
        facade.optimize()
    except BudgetExceeded:
        LOG.info("smac: budget exhausted at %d/%d evals", ctx.n_evals, ctx.max_evals)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
