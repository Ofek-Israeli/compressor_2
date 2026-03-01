"""
Optimizer registry for zero-order methods.

Each optimizer module implements:
    run(ctx: EvalContext, x0: List[float], bounds: Tuple[float, float],
        cfg: EvolutionConfig) -> OptimizerResult

See docs/zero_order_opt_plan.md §3.3, §4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..eval_context import EvalContext
    from ..config import EvolutionConfig


@dataclass
class OptimizerResult:
    best_x: List[float]
    best_f: float
    n_evals_used: int
    history_path: Optional[str] = None


OptimizerFn = Callable[
    ["EvalContext", List[float], Tuple[float, float], "EvolutionConfig"],
    OptimizerResult,
]

_REGISTRY: dict[str, str] = {
    "grid_search": "evolution.optimizers.grid_search",
    "random_search": "evolution.optimizers.random_search",
    "spsa": "evolution.optimizers.spsa",
    "random_direction_2pt": "evolution.optimizers.random_direction_2pt",
    "differential_evolution": "evolution.optimizers.differential_evolution",
    "cmaes": "evolution.optimizers.cmaes",
    "optuna_tpe": "evolution.optimizers.optuna_tpe",
    "smac": "evolution.optimizers.smac_opt",
    "tr_dfo": "evolution.optimizers.tr_dfo",
    "skopt": "evolution.optimizers.skopt_opt",
    "hybrid": "evolution.optimizers.hybrid",
}


def get_optimizer(method: str) -> OptimizerFn:
    """Return the ``run`` callable for the given method name."""
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown optimization method {method!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    import importlib
    mod = importlib.import_module(f".{_REGISTRY[method].split('.')[-1]}", __package__)
    return mod.run  # type: ignore[attr-defined]
