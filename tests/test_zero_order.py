"""
Unit and integration tests for the zero-order optimization pipeline.

Tests use a mock EvalContext so no GPU or SGLang is required.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from compressor_2.evolution.config import EvolutionConfig
from compressor_2.evolution.eval_context import BudgetExceeded, EvalContext
from compressor_2.evolution.optimizers import OptimizerResult, get_optimizer


# ---------------------------------------------------------------------------
# Mock EvalContext that uses a quadratic objective (no GPU required)
# ---------------------------------------------------------------------------

class MockEvalContext:
    """Lightweight mock that mirrors EvalContext's public API.

    Objective: f(x) = -sum((xi - target_i)^2) — maximized at target.
    """

    def __init__(
        self,
        d: int,
        max_evals: int,
        target: Optional[List[float]] = None,
    ) -> None:
        self.cluster_ids = [str(i) for i in range(d)]
        self.max_evals = max_evals
        self.n_evals = 0
        self.best_x: Optional[List[float]] = None
        self.best_f: Optional[float] = None
        self.best_eval_id: Optional[int] = None
        self.out_dir = Path(tempfile.mkdtemp())
        self._target = target or [0.0] * d
        self._eval_id = 0

    def evaluate(
        self, x: List[float], next_x: Optional[List[float]] = None,
    ) -> float:
        if self.n_evals >= self.max_evals:
            raise BudgetExceeded("mock budget exceeded")
        self.n_evals += 1
        f = -sum((xi - ti) ** 2 for xi, ti in zip(x, self._target))
        self._eval_id += 1
        if self.best_f is None or f > self.best_f:
            self.best_f = f
            self.best_x = list(x)
            self.best_eval_id = self._eval_id
        return f

    def evaluate_batch(self, xs: List[List[float]]) -> List[float]:
        results = []
        for i, x in enumerate(xs):
            nxt = xs[i + 1] if i + 1 < len(xs) else None
            results.append(self.evaluate(x, next_x=nxt))
        return results

    def shutdown(self) -> None:
        pass


def _make_cfg(**overrides: Any) -> EvolutionConfig:
    """Build a minimal EvolutionConfig for testing."""
    defaults = dict(
        optimization_method="random_search",
        zero_order_max_evals=20,
        eval_minibatch_size=2,
        eval_seed=42,
        eval_timeout_s=10,
        eval_max_retries=0,
        eval_failure_fitness=-1e9,
        deltas_bound_low=-2.0,
        deltas_bound_high=2.0,
        optimizer_seed=0,
        eval_deterministic=True,
        inference_seed=0,
        llm_max_tokens=256,
        enable_cache=False,
        cache_round_decimals=6,
        run_final_full_pool_eval=False,
        grid_low=-1.0,
        grid_high=1.0,
        grid_step=1.0,
        grid_max_combos=100000,
        grid_allow_truncation=False,
        grid_batch_size=8,
        zo_step_size=0.1,
        zo_perturb_scale=0.05,
        zo_num_directions=1,
        zo_t=1.0,
        tr_dfo_method="bobyqa",
        skopt_n_random_starts=5,
        optuna_n_trials=0,
        hybrid_global_method="differential_evolution",
        hybrid_global_evals=0,
        hybrid_local_evals=0,
        lambda_shortness=0.5,
        lambda_correctness=0.5,
    )
    defaults.update(overrides)
    return EvolutionConfig(**defaults)


# ---------------------------------------------------------------------------
# Test: EvalContext budget enforcement
# ---------------------------------------------------------------------------

class TestEvalContextBudget(unittest.TestCase):

    def test_budget_exhaustion(self):
        """BudgetExceeded is raised after max_evals evaluations."""
        d, max_evals = 2, 5
        tmpdir = tempfile.mkdtemp()
        cfg = _make_cfg()
        ctx = EvalContext(
            cluster_ids=[str(i) for i in range(d)],
            evaluation_indices=[0, 1],
            cfg=cfg,
            tokenizer=MagicMock(),
            server_holder={"server": None},
            out_dir=Path(tmpdir),
            sglang_gpu_id="1",
            max_evals=max_evals,
        )
        with patch(
            "compressor_2.evolution.eval_context.evaluate_x",
            return_value=(1.0, []),
        ):
            for _ in range(max_evals):
                ctx.evaluate([0.0] * d)
            with self.assertRaises(BudgetExceeded):
                ctx.evaluate([0.0] * d)

    def test_best_tracking(self):
        """EvalContext tracks the best x/f correctly."""
        d = 2
        tmpdir = tempfile.mkdtemp()
        cfg = _make_cfg()
        ctx = EvalContext(
            cluster_ids=[str(i) for i in range(d)],
            evaluation_indices=[0],
            cfg=cfg,
            tokenizer=MagicMock(),
            server_holder={"server": None},
            out_dir=Path(tmpdir),
            sglang_gpu_id="1",
            max_evals=100,
        )
        values = [1.0, 5.0, 3.0, 7.0, 2.0]
        call_iter = iter(values)

        def _mock_evaluate_x(*args, **kwargs):
            return (next(call_iter), [])

        with patch(
            "compressor_2.evolution.eval_context.evaluate_x",
            side_effect=_mock_evaluate_x,
        ):
            for _ in values:
                ctx.evaluate([0.0] * d)

        self.assertAlmostEqual(ctx.best_f, 7.0)


# ---------------------------------------------------------------------------
# Test: optimizer registry
# ---------------------------------------------------------------------------

class TestOptimizerRegistry(unittest.TestCase):

    def test_known_methods(self):
        for method in [
            "grid_search", "random_search", "spsa", "random_direction_2pt",
            "differential_evolution",
        ]:
            fn = get_optimizer(method)
            self.assertTrue(callable(fn), f"{method} is not callable")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            get_optimizer("nonexistent_method")


# ---------------------------------------------------------------------------
# Test: grid_search optimizer
# ---------------------------------------------------------------------------

class TestGridSearch(unittest.TestCase):

    def test_small_grid(self):
        """Grid search over 2 dims with step=1.0 => 3^2=9 points on [-1, 1]."""
        d = 2
        cfg = _make_cfg(
            optimization_method="grid_search",
            grid_low=-1.0, grid_high=1.0, grid_step=1.0,
            grid_max_combos=100, grid_batch_size=4,
            zero_order_max_evals=100,
        )
        ctx = MockEvalContext(d=d, max_evals=100, target=[0.0, 0.0])
        fn = get_optimizer("grid_search")
        result = fn(ctx, [0.0, 0.0], (-1.0, 1.0), cfg)

        self.assertEqual(ctx.n_evals, 9)
        self.assertAlmostEqual(result.best_f, 0.0)
        self.assertEqual(result.best_x, [0.0, 0.0])

    def test_grid_truncation(self):
        """Grid search with truncation stops at max_evals."""
        d = 2
        cfg = _make_cfg(
            optimization_method="grid_search",
            grid_low=-1.0, grid_high=1.0, grid_step=0.5,
            grid_max_combos=100000, grid_allow_truncation=True,
            grid_batch_size=4, zero_order_max_evals=5,
        )
        ctx = MockEvalContext(d=d, max_evals=5, target=[0.0, 0.0])
        fn = get_optimizer("grid_search")
        result = fn(ctx, [0.0, 0.0], (-1.0, 1.0), cfg)

        self.assertEqual(ctx.n_evals, 5)

    def test_grid_combo_limit(self):
        """Grid search fails fast when total_combos > grid_max_combos."""
        d = 3
        cfg = _make_cfg(
            optimization_method="grid_search",
            grid_low=-1.0, grid_high=1.0, grid_step=0.1,
            grid_max_combos=10, zero_order_max_evals=100000,
        )
        ctx = MockEvalContext(d=d, max_evals=100000)
        fn = get_optimizer("grid_search")
        with self.assertRaises(ValueError):
            fn(ctx, [0.0] * d, (-1.0, 1.0), cfg)


# ---------------------------------------------------------------------------
# Test: random_search optimizer
# ---------------------------------------------------------------------------

class TestRandomSearch(unittest.TestCase):

    def test_basic(self):
        d, budget = 3, 20
        cfg = _make_cfg(
            optimization_method="random_search",
            zero_order_max_evals=budget,
        )
        ctx = MockEvalContext(d=d, max_evals=budget, target=[0.5, 0.5, 0.5])
        fn = get_optimizer("random_search")
        result = fn(ctx, [0.0] * d, (-2.0, 2.0), cfg)

        self.assertEqual(ctx.n_evals, budget)
        self.assertIsNotNone(result.best_x)
        self.assertGreater(result.best_f, -float("inf"))


# ---------------------------------------------------------------------------
# Test: SPSA optimizer
# ---------------------------------------------------------------------------

class TestSPSA(unittest.TestCase):

    def test_basic(self):
        d, budget = 2, 20
        cfg = _make_cfg(
            optimization_method="spsa",
            zero_order_max_evals=budget,
            zo_step_size=0.1, zo_perturb_scale=0.05,
        )
        ctx = MockEvalContext(d=d, max_evals=budget, target=[0.0, 0.0])
        fn = get_optimizer("spsa")
        result = fn(ctx, [1.0, 1.0], (-2.0, 2.0), cfg)

        self.assertEqual(ctx.n_evals, budget)
        self.assertIsNotNone(result.best_f)


# ---------------------------------------------------------------------------
# Test: random_direction_2pt optimizer
# ---------------------------------------------------------------------------

class TestRandomDirection2pt(unittest.TestCase):

    def test_basic(self):
        d, budget = 2, 20
        cfg = _make_cfg(
            optimization_method="random_direction_2pt",
            zero_order_max_evals=budget,
        )
        ctx = MockEvalContext(d=d, max_evals=budget, target=[0.0, 0.0])
        fn = get_optimizer("random_direction_2pt")
        result = fn(ctx, [1.0, 1.0], (-2.0, 2.0), cfg)

        self.assertEqual(ctx.n_evals, budget)


# ---------------------------------------------------------------------------
# Test: differential_evolution optimizer
# ---------------------------------------------------------------------------

class TestDifferentialEvolution(unittest.TestCase):

    def test_basic(self):
        d, budget = 2, 50
        cfg = _make_cfg(
            optimization_method="differential_evolution",
            zero_order_max_evals=budget,
        )
        ctx = MockEvalContext(d=d, max_evals=budget, target=[0.0, 0.0])
        fn = get_optimizer("differential_evolution")
        result = fn(ctx, [1.0, 1.0], (-2.0, 2.0), cfg)

        self.assertLessEqual(ctx.n_evals, budget)
        self.assertIsNotNone(result.best_f)


# ---------------------------------------------------------------------------
# Test: kconfig_loader validation
# ---------------------------------------------------------------------------

class TestKconfigLoaderValidation(unittest.TestCase):

    def test_validate_zero_order_grid_step(self):
        from compressor_2.kconfig_loader import _validate_zero_order_options

        cfg = _make_cfg(optimization_method="grid_search", grid_step=-0.1)
        with self.assertRaises(ValueError):
            _validate_zero_order_options(cfg)

    def test_validate_zero_order_bounds(self):
        from compressor_2.kconfig_loader import _validate_zero_order_options

        cfg = _make_cfg(
            optimization_method="random_search",
            deltas_bound_low=2.0, deltas_bound_high=1.0,
        )
        with self.assertRaises(ValueError):
            _validate_zero_order_options(cfg)

    def test_validate_skopt_determinism(self):
        from compressor_2.kconfig_loader import _validate_zero_order_options

        cfg = _make_cfg(optimization_method="skopt", eval_deterministic=False)
        with self.assertRaises(ValueError):
            _validate_zero_order_options(cfg)

    def test_validate_grid_search_determinism(self):
        from compressor_2.kconfig_loader import _validate_zero_order_options

        cfg = _make_cfg(optimization_method="grid_search", eval_deterministic=False)
        with self.assertRaises(ValueError):
            _validate_zero_order_options(cfg)


if __name__ == "__main__":
    unittest.main()
