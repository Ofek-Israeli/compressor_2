"""
Optuna TPE (Tree-structured Parzen Estimator) optimizer.

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
        import optuna
    except ImportError:
        raise ImportError(
            "optuna_tpe method requires the 'optuna' package. "
            "Install with: pip install optuna"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    d = len(ctx.cluster_ids)
    lo, hi = bounds
    n_trials = cfg.optuna_n_trials if cfg.optuna_n_trials > 0 else ctx.max_evals

    LOG.info("optuna_tpe: d=%d, bounds=[%.4f, %.4f], n_trials=%d", d, lo, hi, n_trials)

    sampler = optuna.samplers.TPESampler(seed=cfg.optimizer_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.enqueue_trial({f"x{i}": x0[i] for i in range(d)})

    def _objective(trial: optuna.Trial) -> float:
        x = [trial.suggest_float(f"x{i}", lo, hi) for i in range(d)]
        return ctx.evaluate(x)

    try:
        study.optimize(_objective, n_trials=n_trials)
    except BudgetExceeded:
        LOG.info("optuna_tpe: budget exhausted at %d/%d evals", ctx.n_evals, ctx.max_evals)

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
