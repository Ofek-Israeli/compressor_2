"""
Zero-order optimization driver — entry point for all methods except DEAP.

Mirrors ga_driver.run_ga_evolution: validates 2 GPUs, loads data, starts
SGLang once, builds EvalContext, dispatches to the chosen optimizer, and
persists results.

See docs/zero_order_opt_plan.md §3.4, §12.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import EvolutionConfig
from .eval_context import EvalContext
from .graph import update_coord_rd_grad_stats_graph, update_zero_order_fitness_graph
from .gpu_utils import validate_2_gpus, verify_gpu_pinning
from .objective import (
    deltas_dict_to_list,
    deltas_list_to_dict,
    evaluate_x,
    generate_processor,
    load_json,
    save_json,
)
from .optimizers import get_optimizer
from .sglang_lifecycle import SGLangServer

LOG = logging.getLogger(__name__)


def _validate_grid_combos(cfg: EvolutionConfig, d: int) -> None:
    """Validate grid total_combos against limits (requires d from cluster_ids)."""
    n_points = int(math.floor((cfg.grid_high - cfg.grid_low) / cfg.grid_step)) + 1
    total_combos = n_points ** d
    LOG.info(
        "grid_search validation: n_points=%d, d=%d, total_combos=%d",
        n_points, d, total_combos,
    )
    if total_combos > cfg.grid_max_combos:
        raise ValueError(
            f"Grid search: total_combos ({total_combos}) exceeds "
            f"GRID_MAX_COMBOS ({cfg.grid_max_combos}). "
            f"n_points={n_points} per dimension, d={d} dimensions (clusters). "
            f"Grid search is exponential in d: use a coarser GRID_STEP (e.g. 1.0 or 2.0), "
            f"or set GRID_ALLOW_TRUNCATION=y and limit ZERO_ORDER_MAX_EVALS to run a partial grid."
        )
    if total_combos > cfg.zero_order_max_evals and not cfg.grid_allow_truncation:
        raise ValueError(
            f"Grid search: total_combos ({total_combos}) exceeds "
            f"max_evals ({cfg.zero_order_max_evals}) and "
            f"GRID_ALLOW_TRUNCATION is off."
        )


def run_zero_order_evolution(cfg: EvolutionConfig) -> None:
    """Run the zero-order optimization loop (for any method != 'deap')."""

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2-GPU requirement ----
    sglang_gpu_id = validate_2_gpus()
    LOG.info(
        "2-GPU zero-order: embedding=cuda:0, SGLang shim GPU=%s, method=%s",
        sglang_gpu_id, cfg.optimization_method,
    )

    # ---- Load data ----
    initial_deltas: Dict[str, float] = load_json(cfg.initial_deltas_path)
    cluster_ids = sorted(initial_deltas.keys(), key=lambda x: int(x))
    d = len(cluster_ids)
    cfg.expected_cluster_ids = cluster_ids
    LOG.info("Cluster IDs (%d): %s", d, cluster_ids)

    # ---- Grid search: validate combos (needs d) ----
    if cfg.optimization_method == "grid_search":
        _validate_grid_combos(cfg, d)

    # ---- Example pool and fixed minibatch ----
    from financebench_runner.data import load_financebench

    all_examples = load_financebench(cfg.financebench_jsonl)
    if cfg.pool_indices is not None:
        pool_indices = [i for i in cfg.pool_indices if 0 <= i < len(all_examples)]
    else:
        pool_indices = list(range(len(all_examples)))

    rng = random.Random(cfg.eval_seed)
    if cfg.eval_minibatch_size >= len(pool_indices):
        evaluation_indices = list(pool_indices)
    else:
        evaluation_indices = sorted(rng.sample(pool_indices, cfg.eval_minibatch_size))

    indices_hash = hashlib.sha256(
        json.dumps(sorted(evaluation_indices)).encode()
    ).hexdigest()[:16]

    save_json(
        {"indices": evaluation_indices, "seed": cfg.eval_seed, "hash": indices_hash},
        str(out_dir / "evaluation_indices.json"),
    )
    LOG.info(
        "Fixed minibatch: %d examples, seed=%d, hash=%s",
        len(evaluation_indices), cfg.eval_seed, indices_hash,
    )

    # ---- Tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.sglang_model_path)

    # ---- Start SGLang once ----
    server_holder: Dict[str, Any] = {"server": None}
    srv = SGLangServer(cfg, sglang_gpu_id=sglang_gpu_id)
    srv.start()
    server_holder["server"] = srv

    if srv._proc is not None:
        time.sleep(2)
        verify_gpu_pinning(srv._proc.pid)
    LOG.info("SGLang started once; kept running across all evaluations")

    # ---- Plot callback: accumulate (eval_id, f, best_f) and update PNG every N evals ----
    plot_entries: List[Tuple[int, Optional[float], Optional[float]]] = []
    PLOT_UPDATE_INTERVAL = 10

    def _on_eval_done(
        eval_id: int,
        f: Optional[float],
        best_f: Optional[float],
        n_evals: int,
    ) -> None:
        plot_entries.append((eval_id, f, best_f))
        if n_evals > 0 and n_evals % PLOT_UPDATE_INTERVAL == 0:
            try:
                update_zero_order_fitness_graph(
                    plot_entries, str(out_dir), cfg.optimization_method,
                )
            except Exception:
                LOG.debug("Plot update failed (non-fatal)", exc_info=True)

    # ---- Build EvalContext ----
    ctx = EvalContext(
        cluster_ids=cluster_ids,
        evaluation_indices=evaluation_indices,
        cfg=cfg,
        tokenizer=tokenizer,
        server_holder=server_holder,
        out_dir=out_dir,
        sglang_gpu_id=sglang_gpu_id,
        max_evals=cfg.zero_order_max_evals,
        on_eval_done=_on_eval_done,
    )

    # ---- Initial point ----
    x0 = deltas_dict_to_list(initial_deltas, cluster_ids)
    lo, hi = cfg.deltas_bound_low, cfg.deltas_bound_high
    x0 = [max(lo, min(hi, xi)) for xi in x0]
    bounds = (lo, hi)

    # ---- Dispatch ----
    optimizer_fn = get_optimizer(cfg.optimization_method)

    # ---- Resume (load state if present) ----
    state_path = out_dir / "zero_order_state.json"
    if state_path.exists():
        try:
            prev = load_json(str(state_path))
            prev_method = prev.get("method")
            prev_hash = prev.get("indices_hash")
            if prev_method == cfg.optimization_method and prev_hash == indices_hash:
                n_used = prev.get("n_evals_used", 0)
                prev_best_x = prev.get("best_x")
                prev_best_f = prev.get("best_f")
                if prev_best_x is not None:
                    x0 = [max(lo, min(hi, xi)) for xi in prev_best_x]
                    ctx.best_x = list(prev_best_x)
                    ctx.best_f = prev_best_f
                ctx.n_evals = n_used
                LOG.info(
                    "Resumed from zero_order_state.json: n_evals=%d, best_f=%s",
                    n_used, prev_best_f,
                )
            else:
                LOG.info(
                    "State file found but method/hash mismatch; starting fresh"
                )
        except Exception:
            LOG.warning("Failed to load zero_order_state.json; starting fresh")

    # ---- SIGINT handler ----
    interrupted = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        LOG.warning("SIGINT received — persisting state and exiting …")

    signal.signal(signal.SIGINT, _sigint_handler)

    result = None
    try:
        LOG.info(
            "Starting %s: d=%d, budget=%d (remaining=%d)",
            cfg.optimization_method, d, cfg.zero_order_max_evals,
            cfg.zero_order_max_evals - ctx.n_evals,
        )
        result = optimizer_fn(ctx, x0, bounds, cfg)

        LOG.info(
            "%s done: best_f=%.6f, n_evals=%d",
            cfg.optimization_method,
            result.best_f,
            result.n_evals_used,
        )

    finally:
        signal.signal(signal.SIGINT, original_sigint)

        # ---- Persist state ----
        state: Dict[str, Any] = {
            "method": cfg.optimization_method,
            "n_evals_used": ctx.n_evals,
            "max_evals": ctx.max_evals,
            "remaining_evals": max(0, ctx.max_evals - ctx.n_evals),
            "best_x": ctx.best_x,
            "best_f": ctx.best_f,
            "best_eval_id": ctx.best_eval_id,
            "optimizer_seed": cfg.optimizer_seed,
            "eval_seed": cfg.eval_seed,
            "indices_hash": indices_hash,
            "history_path": str(out_dir / "zero_order_history.jsonl"),
        }
        if cfg.optimization_method == "grid_search":
            n_pts = int(math.floor((cfg.grid_high - cfg.grid_low) / cfg.grid_step)) + 1
            state.update({
                "d": d,
                "grid_low": cfg.grid_low,
                "grid_high": cfg.grid_high,
                "grid_step": cfg.grid_step,
                "n_points": n_pts,
                "total_combos": n_pts ** d,
                "truncation": cfg.grid_allow_truncation,
            })

        if cfg.optimization_method == "coordinate_then_random_direction":
            ms = ctx.get_method_state("coord_rd")
            if ms is not None:
                state.update({
                    "last_mean_abs_grad_basis": ms.get("mean_abs_grad_basis"),
                    "last_mean_abs_grad_rand": ms.get("mean_abs_grad_rand"),
                    "n_basis_pairs": ms.get("n_basis_pairs", 0),
                    "n_rand_pairs": ms.get("n_rand_pairs", 0),
                    "coord_alpha_current": ms.get("coord_alpha_current"),
                    "rand_alpha_current": ms.get("rand_alpha_current"),
                    "current_alpha": ms.get("current_alpha"),
                    "current_phase": ms.get("current_phase"),
                    "current_iteration": ms.get("current_iteration"),
                    "coord_iter": ms.get("coord_iter"),
                    "rand_iter": ms.get("rand_iter"),
                    "coord_k": ms.get("coord_k"),
                    "coord_alpha0": ms.get("coord_alpha0"),
                    "coord_alpha_min": ms.get("coord_alpha_min"),
                    "coord_shrink": ms.get("coord_shrink"),
                    "rand_alpha0": ms.get("rand_alpha0"),
                    "rand_alpha_min": ms.get("rand_alpha_min"),
                    "rand_shrink": ms.get("rand_shrink"),
                    "rand_dir_dist": ms.get("rand_dir_dist"),
                    "rand_dirs_per_iter": ms.get("rand_dirs_per_iter"),
                })

        save_json(state, str(state_path))

        # ---- Update zero-order fitness graph (final) ----
        if plot_entries:
            try:
                update_zero_order_fitness_graph(
                    plot_entries, str(out_dir), cfg.optimization_method,
                )
            except Exception:
                LOG.debug("Final plot update failed (non-fatal)", exc_info=True)

        if cfg.optimization_method == "coordinate_then_random_direction":
            try:
                update_coord_rd_grad_stats_graph(
                    str(out_dir / "zero_order_history.jsonl"), str(out_dir),
                )
            except Exception:
                LOG.debug("Grad stats plot update failed (non-fatal)", exc_info=True)

        if ctx.best_x is not None:
            best_deltas = deltas_list_to_dict(ctx.best_x, cluster_ids)
            save_json(best_deltas, str(out_dir / "deltas_best.json"))
            LOG.info("Best-ever fitness: %.6f", ctx.best_f or 0)
        else:
            LOG.warning("No best individual recorded")

        # ---- Optional final full-pool evaluation ----
        if cfg.run_final_full_pool_eval and ctx.best_x is not None:
            LOG.info("Running final full-pool evaluation …")
            try:
                full_indices = list(range(len(all_examples)))
                f_full, _ = evaluate_x(
                    ctx.best_x, cluster_ids, full_indices, cfg,
                    tokenizer, server_holder, out_dir, sglang_gpu_id,
                )
                final_payload: Dict[str, Any] = {
                    "best_x": ctx.best_x, "f_full_pool": f_full,
                }
                if cfg.optimization_method == "coordinate_then_random_direction":
                    ms = ctx.get_method_state("coord_rd")
                    src = ms if ms else (result.method_summary if result and result.method_summary else None)
                    if src:
                        final_payload.update({
                            "final_mean_abs_grad_basis": src.get("mean_abs_grad_basis"),
                            "final_mean_abs_grad_rand": src.get("mean_abs_grad_rand"),
                            "final_n_basis_pairs": src.get("n_basis_pairs", 0),
                            "final_n_rand_pairs": src.get("n_rand_pairs", 0),
                        })
                save_json(final_payload, str(out_dir / "final_eval.json"))
                state["final_full_pool_f"] = f_full
                save_json(state, str(state_path))
                LOG.info("Full-pool fitness: %.6f", f_full)
            except Exception:
                LOG.exception("Final full-pool evaluation failed (non-fatal)")

        # ---- Generate processor_best.py once at end ----
        if ctx.best_x is not None:
            try:
                best_deltas_path = str(out_dir / "deltas_best.json")
                processor_best_path = str(out_dir / "processor_best.py")
                generate_processor(cfg, best_deltas_path, processor_best_path)
                LOG.info("Generated processor_best.py")
            except Exception:
                LOG.exception("Failed to generate processor_best.py (non-fatal)")

        # ---- Cleanup ----
        ctx.shutdown()

        srv_ref = server_holder.get("server")
        if srv_ref is not None:
            try:
                srv_ref.stop()
            except Exception:
                LOG.exception("Error stopping SGLang during cleanup")

        LOG.info("Zero-order outputs in %s", cfg.output_dir)
