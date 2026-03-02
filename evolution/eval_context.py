"""
EvalContext — shared evaluation context for zero-order optimizers.

Wraps the shared objective (evolution/objective.py) with budget tracking,
prefetch, optional caching, failure handling, and JSONL history logging.
All zero-order optimizers must evaluate through this context.

See docs/zero_order_opt_plan.md §3.2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import EvolutionConfig
from .objective import (
    deltas_list_to_dict,
    evaluate_x,
    generate_processor,
    save_json,
)

LOG = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Raised when the evaluation budget is exhausted."""


class EvalContext:
    """Manages evaluation state, prefetch, budget, caching, and history.

    All zero-order optimizers must call ``evaluate`` or ``evaluate_batch``
    rather than calling the objective directly.
    """

    def __init__(
        self,
        cluster_ids: List[str],
        evaluation_indices: List[int],
        cfg: EvolutionConfig,
        tokenizer: Any,
        server_holder: Dict[str, Any],
        out_dir: Path,
        sglang_gpu_id: str,
        max_evals: int,
        on_eval_done: Optional[
            Callable[[int, Optional[float], Optional[float], int], None]
        ] = None,
    ) -> None:
        self.cluster_ids = cluster_ids
        self.evaluation_indices = evaluation_indices
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.server_holder = server_holder
        self.out_dir = out_dir
        self.sglang_gpu_id = sglang_gpu_id
        self.max_evals = max_evals
        self._on_eval_done = on_eval_done

        self.n_evals = 0
        self.best_x: Optional[List[float]] = None
        self.best_f: Optional[float] = None
        self.best_eval_id: Optional[int] = None
        self._eval_id_counter = 0

        self.indices_hash = hashlib.sha256(
            json.dumps(sorted(evaluation_indices)).encode()
        ).hexdigest()[:16]

        self._prefetch_counter = 0
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="zo_prefetch",
        )
        self._prefetch_future: Optional[Future[Optional[str]]] = None

        self._cache: Dict[Tuple, float] = {}
        self._cache_enabled = cfg.eval_deterministic and cfg.enable_cache

        self._history_path = str(out_dir / "zero_order_history.jsonl")

        self._extra_log_fields: Dict[str, Any] = {}
        self._method_states: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        x: List[float],
        next_x: Optional[List[float]] = None,
    ) -> float:
        """Evaluate a single point.

        Waits for any in-flight prefetch, submits prefetch for *next_x*
        (if provided), then evaluates *x*.  Raises ``BudgetExceeded``
        if the budget is exhausted.
        """
        if self.n_evals >= self.max_evals:
            raise BudgetExceeded(
                f"Budget exhausted: {self.n_evals}/{self.max_evals} evals used"
            )

        if self._cache_enabled:
            key = self._cache_key(x)
            if key in self._cache:
                LOG.info("  eval (cached): x hash=%s", key[:8])
                f = self._cache[key]
                self._update_best(x, f)
                return f

        pre_path = self._consume_prefetch()
        self._submit_prefetch(next_x)

        eval_id = self._next_eval_id()
        f = self._run_eval(x, eval_id, pre_path)
        return f

    def evaluate_batch(self, xs: List[List[float]]) -> List[float]:
        """Evaluate a batch of points sequentially with prefetch overlap.

        Prefetches ``xs[i+1]`` while evaluating ``xs[i]``.
        """
        results: List[float] = []
        for i, x in enumerate(xs):
            next_x = xs[i + 1] if i + 1 < len(xs) else None
            f = self.evaluate(x, next_x=next_x)
            results.append(f)
        return results

    def set_extra_log_fields(self, d: Dict[str, Any]) -> None:
        """Set fields merged into every subsequent JSONL eval record."""
        self._extra_log_fields = dict(d)

    def get_extra_log_fields(self) -> Dict[str, Any]:
        """Return the current extra-log-fields dict."""
        return dict(self._extra_log_fields)

    def set_method_state(self, name: str, state: Dict[str, Any]) -> None:
        """Store optimizer-specific state retrievable by the driver."""
        self._method_states[name] = dict(state)

    def get_method_state(self, name: str) -> Optional[Dict[str, Any]]:
        """Return previously stored method state, or None."""
        s = self._method_states.get(name)
        return dict(s) if s is not None else None

    def shutdown(self) -> None:
        """Drain prefetch executor.  Call once at the end of the run."""
        if self._prefetch_future is not None:
            try:
                self._prefetch_future.result(timeout=300)
            except Exception:
                pass
        self._prefetch_executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_eval_id(self) -> int:
        eid = self._eval_id_counter
        self._eval_id_counter += 1
        return eid

    def _run_eval(
        self,
        x: List[float],
        eval_id: int,
        pre_generated_processor_path: Optional[str],
    ) -> float:
        """Execute one evaluation with retry logic and budget accounting."""
        last_error: Optional[Exception] = None
        attempts = 1 + self.cfg.eval_max_retries

        for attempt in range(attempts):
            if self.n_evals >= self.max_evals:
                raise BudgetExceeded(
                    f"Budget exhausted during retry: "
                    f"{self.n_evals}/{self.max_evals} evals used"
                )
            self.n_evals += 1
            t0 = time.monotonic()
            try:
                f, results = evaluate_x(
                    x, self.cluster_ids, self.evaluation_indices,
                    self.cfg, self.tokenizer, self.server_holder,
                    self.out_dir, self.sglang_gpu_id,
                    pre_generated_processor_path=(
                        pre_generated_processor_path if attempt == 0 else None
                    ),
                )
                elapsed = time.monotonic() - t0
                self._log_history(eval_id, x, f, "ok", attempt, elapsed)
                self._update_best(x, f)
                if self._cache_enabled:
                    self._cache[self._cache_key(x)] = f
                if self._on_eval_done is not None:
                    self._on_eval_done(eval_id, f, self.best_f, self.n_evals)
                return f
            except Exception as exc:
                elapsed = time.monotonic() - t0
                last_error = exc
                LOG.warning(
                    "Eval %d attempt %d/%d failed after %.1fs: %s",
                    eval_id, attempt + 1, attempts, elapsed, exc,
                )
                self._log_history(
                    eval_id, x, None, "fail", attempt, elapsed,
                    error=str(exc),
                )

        f = self.cfg.eval_failure_fitness
        LOG.warning(
            "Eval %d: all %d attempts failed; returning failure fitness %.4f",
            eval_id, attempts, f,
        )
        if self._on_eval_done is not None:
            self._on_eval_done(eval_id, f, self.best_f, self.n_evals)
        return f

    def _update_best(self, x: List[float], f: float) -> None:
        if self.best_f is None or f > self.best_f:
            self.best_f = f
            self.best_x = list(x)
            self.best_eval_id = self._eval_id_counter - 1

    def _cache_key(self, x: List[float]) -> Tuple:
        decimals = self.cfg.cache_round_decimals
        rounded = tuple(round(xi, decimals) for xi in x)
        return (rounded, self.indices_hash)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _submit_prefetch(self, next_x: Optional[List[float]]) -> None:
        if next_x is None:
            return
        c = self._prefetch_counter
        self._prefetch_counter += 1
        deltas_path = str(self.out_dir / f"_eval_deltas_prefetch_{c}.json")
        processor_path = str(self.out_dir / f"_eval_processor_prefetch_{c}.py")
        save_json(
            deltas_list_to_dict(next_x, self.cluster_ids), deltas_path,
        )
        self._prefetch_future = self._prefetch_executor.submit(
            self._run_prefetch, deltas_path, processor_path,
        )

    def _consume_prefetch(self) -> Optional[str]:
        if self._prefetch_future is None:
            return None
        try:
            result = self._prefetch_future.result(timeout=300)
        except Exception:
            LOG.exception("Prefetch wait failed")
            result = None
        self._prefetch_future = None
        return result

    def _run_prefetch(self, deltas_path: str, processor_path: str) -> Optional[str]:
        try:
            generate_processor(self.cfg, deltas_path, processor_path)
            return processor_path
        except Exception:
            LOG.exception("Prefetch generate-processor failed")
            return None

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _log_history(
        self,
        eval_id: int,
        x: List[float],
        f: Optional[float],
        status: str,
        attempt: int,
        elapsed: float,
        error: Optional[str] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "eval_id": eval_id,
            "x": x,
            "status": status,
            "attempt": attempt,
            "indices_hash": self.indices_hash,
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.time(),
            "n_evals": self.n_evals,
        }
        if f is not None:
            record["f"] = f
        if error is not None:
            record["error"] = error
        if self._extra_log_fields:
            record.update(self._extra_log_fields)
        try:
            with open(self._history_path, "a") as fp:
                fp.write(json.dumps(record) + "\n")
        except Exception:
            LOG.warning("Failed to write history entry for eval %d", eval_id)
