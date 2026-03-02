"""
Coordinate-then-random-direction zero-order optimizer.

Phase 1: standard-basis (coordinate) search for K iterations.
Phase 2: random unit-direction search until budget exhausted.
Each phase has independent alpha0/alpha_min/shrink/improvement_eps.

Every +/- pair is evaluated via ctx.evaluate_batch([x_plus, x_minus]).
See docs/coordinates_search_plan.md for full specification.
"""

from __future__ import annotations

import logging
import math
import random as _random
from typing import Any, Dict, List, Optional, Tuple

from ..config import EvolutionConfig
from ..eval_context import BudgetExceeded, EvalContext
from . import OptimizerResult

LOG = logging.getLogger(__name__)


def _clip(x: List[float], lo: float, hi: float) -> List[float]:
    return [max(lo, min(hi, xi)) for xi in x]


def _sample_unit_gaussian(d: int, rng: _random.Random) -> List[float]:
    raw = [rng.gauss(0, 1) for _ in range(d)]
    norm = math.sqrt(sum(r * r for r in raw)) or 1.0
    return [r / norm for r in raw]


def _sample_unit_rademacher(d: int, rng: _random.Random) -> List[float]:
    raw = [rng.choice([-1.0, 1.0]) for _ in range(d)]
    norm = math.sqrt(float(d))
    return [r / norm for r in raw]


def _sample_unit_direction(
    d: int, dist: str, rng: _random.Random,
) -> List[float]:
    if dist == "rademacher_unit":
        return _sample_unit_rademacher(d, rng)
    return _sample_unit_gaussian(d, rng)


def _is_failure_fitness(f: float, cfg: EvolutionConfig) -> bool:
    return f <= cfg.eval_failure_fitness


class _Stats:
    """Running mean of |derivative| for one phase."""

    __slots__ = ("sum_abs", "count")

    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.count = 0

    def update(self, g_hat: float) -> None:
        self.sum_abs += abs(g_hat)
        self.count += 1

    @property
    def mean(self) -> Optional[float]:
        return self.sum_abs / self.count if self.count > 0 else None


def _build_extra_fields(
    basis_stats: _Stats,
    rand_stats: _Stats,
    coord_alpha: float,
    rand_alpha: float,
    phase: str,
    iteration: int,
) -> Dict[str, Any]:
    active_alpha = coord_alpha if phase == "coordinate" else rand_alpha
    return {
        "mean_abs_grad_basis": basis_stats.mean,
        "mean_abs_grad_rand": rand_stats.mean,
        "n_basis_pairs": basis_stats.count,
        "n_rand_pairs": rand_stats.count,
        "current_alpha": active_alpha,
        "coord_alpha": coord_alpha,
        "rand_alpha": rand_alpha,
        "current_phase": phase,
        "current_iteration": iteration,
    }


def _build_method_state(
    basis_stats: _Stats,
    rand_stats: _Stats,
    coord_alpha: float,
    rand_alpha: float,
    phase: str,
    coord_iter: int,
    rand_iter: int,
    cfg: EvolutionConfig,
) -> Dict[str, Any]:
    return {
        "mean_abs_grad_basis": basis_stats.mean,
        "mean_abs_grad_rand": rand_stats.mean,
        "n_basis_pairs": basis_stats.count,
        "n_rand_pairs": rand_stats.count,
        "coord_alpha_current": coord_alpha,
        "rand_alpha_current": rand_alpha,
        "current_alpha": coord_alpha if phase == "coordinate" else rand_alpha,
        "current_phase": phase,
        "current_iteration": coord_iter if phase == "coordinate" else rand_iter,
        "coord_iter": coord_iter,
        "rand_iter": rand_iter,
        # Hyperparams for reproducibility
        "coord_k": cfg.coord_rd_coord_k,
        "coord_alpha0": cfg.coord_rd_coord_alpha0,
        "coord_alpha_min": cfg.coord_rd_coord_alpha_min,
        "coord_shrink": cfg.coord_rd_coord_shrink,
        "coord_num_coords_per_iter": cfg.coord_rd_coord_num_coords_per_iter,
        "coord_opportunistic": cfg.coord_rd_coord_opportunistic,
        "coord_improvement_eps": cfg.coord_rd_coord_improvement_eps,
        "rand_alpha0": cfg.coord_rd_rand_alpha0,
        "rand_alpha_min": cfg.coord_rd_rand_alpha_min,
        "rand_shrink": cfg.coord_rd_rand_shrink,
        "rand_dir_dist": cfg.coord_rd_rand_dir_dist,
        "rand_dirs_per_iter": cfg.coord_rd_rand_dirs_per_iter,
        "rand_improvement_eps": cfg.coord_rd_rand_improvement_eps,
    }


def _sync_state(
    ctx: EvalContext,
    basis_stats: _Stats,
    rand_stats: _Stats,
    coord_alpha: float,
    rand_alpha: float,
    phase: str,
    coord_iter: int,
    rand_iter: int,
    cfg: EvolutionConfig,
) -> None:
    """Push extra_log_fields and method_state to EvalContext."""
    iteration = coord_iter if phase == "coordinate" else rand_iter
    ctx.set_extra_log_fields(
        _build_extra_fields(basis_stats, rand_stats, coord_alpha, rand_alpha, phase, iteration)
    )
    ctx.set_method_state(
        "coord_rd",
        _build_method_state(basis_stats, rand_stats, coord_alpha, rand_alpha, phase, coord_iter, rand_iter, cfg),
    )


def run(
    ctx: EvalContext,
    x0: List[float],
    bounds: Tuple[float, float],
    cfg: EvolutionConfig,
) -> OptimizerResult:
    d = len(ctx.cluster_ids)
    lo, hi = bounds
    rng = _random.Random(cfg.optimizer_seed)

    basis_stats = _Stats()
    rand_stats = _Stats()
    coord_alpha = cfg.coord_rd_coord_alpha0
    rand_alpha = cfg.coord_rd_rand_alpha0
    n_coord_pairs_done = 0
    n_rand_pairs_done = 0

    num_coords = cfg.coord_rd_coord_num_coords_per_iter
    if num_coords == 0 or num_coords >= d:
        num_coords = d

    LOG.info(
        "coordinate_then_random_direction: d=%d, K=%d, "
        "coord_alpha0=%.6f, rand_alpha0=%.6f, budget=%d",
        d, cfg.coord_rd_coord_k, coord_alpha, rand_alpha, ctx.max_evals,
    )

    x = list(x0)
    current_f: Optional[float] = None

    def _remaining() -> int:
        return ctx.max_evals - ctx.n_evals

    try:
        # --- Initial evaluation of x0 ---
        _sync_state(ctx, basis_stats, rand_stats, coord_alpha, rand_alpha, "coordinate", 0, 0, cfg)
        current_f = ctx.evaluate(x0)
        x = list(ctx.best_x) if ctx.best_x is not None else list(x0)
        current_f = ctx.best_f if ctx.best_f is not None else current_f

        # =============================================================
        # Phase 1 — Coordinate (standard-basis) search
        # =============================================================
        for coord_iter in range(cfg.coord_rd_coord_k):
            if _remaining() < 2:
                raise BudgetExceeded("Insufficient budget for next pair")

            # Choose coordinates for this iteration
            if num_coords >= d:
                coords = list(range(d))
                if cfg.coord_rd_coord_shuffle_each_iter:
                    rng.shuffle(coords)
            else:
                if cfg.coord_rd_coord_sample_with_replacement:
                    coords = [rng.randrange(d) for _ in range(num_coords)]
                else:
                    coords = rng.sample(range(d), num_coords)

            best_f_iter = current_f
            best_x_iter = list(x)
            improved_this_iter = False

            for i in coords:
                if _remaining() < 2:
                    raise BudgetExceeded("Insufficient budget for next pair")
                if cfg.coord_rd_coord_max_coords_total > 0 and n_coord_pairs_done >= cfg.coord_rd_coord_max_coords_total:
                    break

                e_i = [0.0] * d
                e_i[i] = 1.0
                x_plus = _clip([xi + coord_alpha * ei for xi, ei in zip(x, e_i)], lo, hi)
                x_minus = _clip([xi - coord_alpha * ei for xi, ei in zip(x, e_i)], lo, hi)

                f_plus, f_minus = ctx.evaluate_batch([x_plus, x_minus])
                n_coord_pairs_done += 1

                if not (_is_failure_fitness(f_plus, cfg) or _is_failure_fitness(f_minus, cfg)):
                    g_hat = (f_plus - f_minus) / (2.0 * coord_alpha)
                    basis_stats.update(g_hat)

                _sync_state(ctx, basis_stats, rand_stats, coord_alpha, rand_alpha, "coordinate", coord_iter, 0, cfg)

                for cand, f_cand in [(x_plus, f_plus), (x_minus, f_minus)]:
                    if best_f_iter is not None and f_cand > best_f_iter + cfg.coord_rd_coord_improvement_eps:
                        best_f_iter = f_cand
                        best_x_iter = list(cand)
                        improved_this_iter = True

                if cfg.coord_rd_coord_opportunistic and improved_this_iter:
                    break

            if improved_this_iter:
                x = best_x_iter
                current_f = best_f_iter
            else:
                coord_alpha = max(cfg.coord_rd_coord_alpha_min, coord_alpha * cfg.coord_rd_coord_shrink)

            _sync_state(ctx, basis_stats, rand_stats, coord_alpha, rand_alpha, "coordinate", coord_iter, 0, cfg)

            if cfg.coord_rd_coord_max_coords_total > 0 and n_coord_pairs_done >= cfg.coord_rd_coord_max_coords_total:
                LOG.info("Phase 1: coord_max_coords_total cap reached (%d)", n_coord_pairs_done)
                break

        # =============================================================
        # Phase boundary: reset alpha to RAND_ALPHA0
        # =============================================================
        rand_alpha = cfg.coord_rd_rand_alpha0
        _sync_state(ctx, basis_stats, rand_stats, coord_alpha, rand_alpha, "random_direction", 0, 0, cfg)
        LOG.info(
            "Phase switch: coordinate -> random_direction, "
            "coord_alpha=%.6f, rand_alpha=%.6f, evals_used=%d",
            coord_alpha, rand_alpha, ctx.n_evals,
        )

        # Refresh current best from context
        if ctx.best_x is not None:
            x = list(ctx.best_x)
            current_f = ctx.best_f

        # =============================================================
        # Phase 2 — Random-direction search
        # =============================================================
        rand_iter = 0
        while True:
            if _remaining() < 2:
                raise BudgetExceeded("Insufficient budget for next pair")

            best_f_iter = current_f
            best_x_iter = list(x)
            x_start = list(x)
            improved_this_iter = False

            for _ in range(cfg.coord_rd_rand_dirs_per_iter):
                if _remaining() < 2:
                    raise BudgetExceeded("Insufficient budget for next pair")
                if cfg.coord_rd_rand_max_dir_pairs_total > 0 and n_rand_pairs_done >= cfg.coord_rd_rand_max_dir_pairs_total:
                    break

                u = _sample_unit_direction(d, cfg.coord_rd_rand_dir_dist, rng)
                x_plus = _clip([xi + rand_alpha * ui for xi, ui in zip(x, u)], lo, hi)
                x_minus = _clip([xi - rand_alpha * ui for xi, ui in zip(x, u)], lo, hi)

                f_plus, f_minus = ctx.evaluate_batch([x_plus, x_minus])
                n_rand_pairs_done += 1

                if not (_is_failure_fitness(f_plus, cfg) or _is_failure_fitness(f_minus, cfg)):
                    g_hat = (f_plus - f_minus) / (2.0 * rand_alpha)
                    rand_stats.update(g_hat)

                _sync_state(ctx, basis_stats, rand_stats, coord_alpha, rand_alpha, "random_direction", 0, rand_iter, cfg)

                for cand, f_cand in [(x_plus, f_plus), (x_minus, f_minus)]:
                    if best_f_iter is not None and f_cand > best_f_iter + cfg.coord_rd_rand_improvement_eps:
                        best_f_iter = f_cand
                        best_x_iter = list(cand)
                        improved_this_iter = True

                if cfg.coord_rd_rand_use_current_x_for_next_dir and improved_this_iter:
                    x = best_x_iter
                    current_f = best_f_iter

            if improved_this_iter:
                x = best_x_iter
                current_f = best_f_iter
            else:
                rand_alpha = max(cfg.coord_rd_rand_alpha_min, rand_alpha * cfg.coord_rd_rand_shrink)

            _sync_state(ctx, basis_stats, rand_stats, coord_alpha, rand_alpha, "random_direction", 0, rand_iter, cfg)
            rand_iter += 1

            if cfg.coord_rd_rand_max_dir_pairs_total > 0 and n_rand_pairs_done >= cfg.coord_rd_rand_max_dir_pairs_total:
                LOG.info("Phase 2: rand_max_dir_pairs_total cap reached (%d)", n_rand_pairs_done)
                break

    except BudgetExceeded:
        LOG.info(
            "coordinate_then_random_direction: budget exhausted at %d/%d evals",
            ctx.n_evals, ctx.max_evals,
        )

    summary = {
        "mean_abs_grad_basis": basis_stats.mean,
        "mean_abs_grad_rand": rand_stats.mean,
        "n_basis_pairs": basis_stats.count,
        "n_rand_pairs": rand_stats.count,
    }

    return OptimizerResult(
        best_x=ctx.best_x or x0,
        best_f=ctx.best_f if ctx.best_f is not None else float("-inf"),
        n_evals_used=ctx.n_evals,
        history_path=str(ctx.out_dir / "zero_order_history.jsonl"),
        method_summary=summary,
    )
