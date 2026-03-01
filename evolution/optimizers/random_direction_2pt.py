"""
Random-direction two-point gradient estimator.

Like SPSA but uses a random unit direction instead of Bernoulli.
Each iteration evaluates the ± pair via ``ctx.evaluate_batch``.

See docs/zero_order_opt_plan.md §4, §9.
"""

from __future__ import annotations

import logging
import math
import random as _random
from typing import List, Tuple

from ..config import EvolutionConfig
from ..eval_context import BudgetExceeded, EvalContext
from . import OptimizerResult

LOG = logging.getLogger(__name__)


def _clip(x: List[float], lo: float, hi: float) -> List[float]:
    return [max(lo, min(hi, xi)) for xi in x]


def _random_unit(d: int, rng: _random.Random) -> List[float]:
    raw = [rng.gauss(0, 1) for _ in range(d)]
    norm = math.sqrt(sum(r * r for r in raw)) or 1.0
    return [r / norm for r in raw]


def run(
    ctx: EvalContext,
    x0: List[float],
    bounds: Tuple[float, float],
    cfg: EvolutionConfig,
) -> OptimizerResult:
    d = len(ctx.cluster_ids)
    lo, hi = bounds
    a = cfg.zo_step_size
    c = cfg.zo_perturb_scale
    rng = _random.Random(cfg.optimizer_seed)

    x = list(x0)

    LOG.info("random_direction_2pt: d=%d, a=%.6f, c=%.6f, budget=%d", d, a, c, ctx.max_evals)

    try:
        while True:
            direction = _random_unit(d, rng)
            x_plus = _clip([xi + c * di for xi, di in zip(x, direction)], lo, hi)
            x_minus = _clip([xi - c * di for xi, di in zip(x, direction)], lo, hi)

            f_plus, f_minus = ctx.evaluate_batch([x_plus, x_minus])

            grad = [(f_plus - f_minus) / (2.0 * c) * di for di in direction]
            x = _clip([xi + a * gi for xi, gi in zip(x, grad)], lo, hi)
    except BudgetExceeded:
        LOG.info(
            "random_direction_2pt: budget exhausted at %d/%d evals",
            ctx.n_evals, ctx.max_evals,
        )

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
    )
