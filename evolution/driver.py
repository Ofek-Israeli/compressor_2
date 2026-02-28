"""
Evolution driver: the main loop that evolves steering deltas.

Orchestrates: stop SGLang → generate-processor → start SGLang →
run minibatch → correctness per example → reflector (all + correctness) →
update deltas → track best by composite score → graph.  Handles SIGINT for graceful exit.
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import EvolutionConfig
from .graph import update_graph
from .reflector import (
    build_reflector_message,
    call_reflector,
    log_reflector_message,
    validate_reflector_output,
)
from .sglang_lifecycle import SGLangServer

LOG = logging.getLogger(__name__)


class _GracefulExit(Exception):
    """Raised by SIGINT handler to trigger graceful shutdown."""


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _generate_processor(cfg: EvolutionConfig, deltas_path: str, output_path: str) -> None:
    """Run generate-processor as a subprocess (needs GPU)."""
    cmd = [
        sys.executable, "-m", "compressor_2", "generate-processor",
        cfg.kmeans_path, deltas_path,
        "-o", output_path,
        "--tokenizer", cfg.sglang_model_path,
        "--embedding-model", cfg.embedding_model,
        "--embeddings", cfg.embeddings_path,
        "-b", str(cfg.gen_batch_size),
    ]
    LOG.info("Running generate-processor: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOG.error("generate-processor stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"generate-processor failed (exit {result.returncode})"
        )
    LOG.info("Processor generated: %s", output_path)


def _run_minibatch(
    runner_config_path: str,
    financebench_jsonl: str,
    processor_path: str,
) -> List[Dict[str, Any]]:
    """Run financebench_runner on a minibatch; return list of result dicts.

    ``runner_config_path`` should already contain the ``example_indices``
    for this minibatch.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp_output = tmp.name

    cmd = [
        sys.executable, "-m", "financebench_runner",
        "--config", runner_config_path,
        "--input", financebench_jsonl,
        "--output", tmp_output,
        "--logit-processor", processor_path,
    ]
    LOG.info("Running financebench_runner...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOG.error("financebench_runner stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"financebench_runner failed (exit {result.returncode})"
        )

    with open(tmp_output, "r") as f:
        results = json.load(f)
    os.unlink(tmp_output)
    return results


def _count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens using the model's tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _run_correctness(
    results: List[Dict[str, Any]],
    cfg: EvolutionConfig,
) -> None:
    """Run correctness evaluation for each result; attach is_correct and correctness_reasoning."""
    from .correctness_openai import evaluate_one

    for r in results:
        ev = evaluate_one(
            predicted=r.get("llm_answer", ""),
            ground_truth=[r.get("ground_truth_answer", "")],
            question=r.get("question", ""),
            model=cfg.correctness_model,
            tolerance=cfg.correctness_tolerance,
            api_key_env=cfg.openai_api_key_env,
        )
        r["is_correct"] = ev.is_correct
        r["correctness_reasoning"] = ev.reasoning


def _sample_minibatch_indices(
    pool: List[int],
    size: int,
) -> List[int]:
    """Randomly sample minibatch indices from the pool."""
    if size >= len(pool):
        return list(pool)
    return random.sample(pool, size)


def _build_runner_config_with_indices(
    base_config_path: str,
    indices: List[int],
    evol_cfg: EvolutionConfig,
) -> str:
    """Create a temp YAML config for the minibatch.

    Overrides example_indices, sglang.base_url, and model_id so the runner
    talks to the SGLang server managed by the evolution driver.
    """
    import yaml

    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["example_indices"] = indices
    cfg["concurrency"] = len(indices)  # run minibatch in parallel
    cfg["model_id"] = evol_cfg.sglang_model_path
    if "sglang" not in cfg or not isinstance(cfg.get("sglang"), dict):
        cfg["sglang"] = {}
    cfg["sglang"]["base_url"] = f"http://localhost:{evol_cfg.sglang_port}"
    cfg["sglang"]["timeout_s"] = evol_cfg.runner_timeout_s

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="evol_runner_"
    )
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def _persist_best(
    cfg: EvolutionConfig,
    best_deltas: Dict[str, float],
) -> None:
    """Save the best deltas to the output directory."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_json(best_deltas, str(out / "deltas_best.json"))
    LOG.info("Persisted deltas_best.json")


def run_evolution(cfg: EvolutionConfig) -> None:
    """Run the full evolution loop."""
    # ---- Setup ----
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_descriptions = _load_json(cfg.cluster_descriptions_path)

    # Determine expected cluster IDs from the initial deltas file
    initial_deltas: Dict[str, float] = _load_json(cfg.initial_deltas_path)
    cfg.expected_cluster_ids = sorted(
        initial_deltas.keys(), key=lambda x: int(x)
    )
    LOG.info("Cluster IDs: %s", cfg.expected_cluster_ids)

    # ---- Resume from existing results if present ----
    history_path = out_dir / "history.json"
    deltas_current_path = out_dir / "deltas_current.json"
    deltas_best_path = out_dir / "deltas_best.json"
    cot_path = out_dir / "cot_summary.txt"

    start_iteration = 0
    cot_summary = ""
    best_composite_score: Optional[float] = None
    best_deltas: Optional[Dict[str, float]] = None
    history: List[Dict[str, Any]] = []

    if history_path.exists():
        raw_history = _load_json(str(history_path))
        # Normalize to new format (composite_score, etc.); support old format
        for h in raw_history:
            if "composite_score" not in h:
                ml = h.get("mean_token_length", 0)
                scale = getattr(cfg, "shortness_scale", 300)
                h = dict(h)
                h["shortness_score"] = 1.0 / (1.0 + ml / scale)
                h["correctness_ratio"] = 0.0
                h["composite_score"] = 0.4 * h["shortness_score"]
            history.append(h)
        if history:
            start_iteration = history[-1]["iteration"] + 1
            LOG.info(
                "Resuming: found %d previous iterations; starting at iteration %d",
                len(history), start_iteration,
            )

    if deltas_current_path.exists() and start_iteration > 0:
        current_deltas = _load_json(str(deltas_current_path))
        LOG.info("Resumed current deltas from %s", deltas_current_path)
    else:
        current_deltas = dict(initial_deltas)
        LOG.info("Initial deltas: %s", current_deltas)

    if deltas_best_path.exists() and start_iteration > 0:
        best_deltas = _load_json(str(deltas_best_path))
        if history:
            best_composite_score = max(h["composite_score"] for h in history)
        LOG.info(
            "Resumed best deltas (best composite_score=%.3f)",
            best_composite_score if best_composite_score is not None else 0,
        )

    if cot_path.exists() and start_iteration > 0:
        cot_summary = cot_path.read_text().strip()
        LOG.info("Resumed evolving summary (%d chars)", len(cot_summary))

    # Load tokenizer for token-length counting (same as SGLang model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.sglang_model_path)

    # Build pool of example indices
    from financebench_runner.data import load_financebench
    all_examples = load_financebench(cfg.financebench_jsonl)
    if cfg.pool_indices is not None:
        pool = [i for i in cfg.pool_indices if 0 <= i < len(all_examples)]
    else:
        pool = list(range(len(all_examples)))
    LOG.info("Example pool size: %d", len(pool))

    server: Optional[SGLangServer] = None
    interrupted = False

    # ---- SIGINT handler ----
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        LOG.warning("SIGINT received; finishing current iteration then exiting...")

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        for iteration in range(start_iteration, cfg.max_iterations):
            if interrupted:
                LOG.info("Interrupted before iteration %d", iteration)
                break

            LOG.info("=== Iteration %d / %d ===", iteration, cfg.max_iterations - 1)

            # 1. Save current deltas
            deltas_path = str(out_dir / "deltas_current.json")
            _save_json(current_deltas, deltas_path)

            # 2. Stop SGLang (if running) so GPU is free for generate-processor
            if server is not None:
                server.stop()
                server = None

            # 3. Generate processor (needs GPU)
            processor_path = str(out_dir / "processor.py")
            _generate_processor(cfg, deltas_path, processor_path)

            # 4. Start SGLang
            server = SGLangServer(cfg, processor_path)
            server.start()

            # 5. Sample minibatch and run
            mb_indices = _sample_minibatch_indices(pool, cfg.minibatch_size)
            LOG.info("Minibatch indices: %s", mb_indices)

            tmp_runner_cfg = _build_runner_config_with_indices(
                cfg.runner_config_path, mb_indices, cfg
            )
            LOG.info(
                "Minibatch: %s examples (indices %s), concurrency=%s",
                len(mb_indices), mb_indices, len(mb_indices),
            )
            try:
                results = _run_minibatch(
                    tmp_runner_cfg, cfg.financebench_jsonl, processor_path
                )
            finally:
                os.unlink(tmp_runner_cfg)

            if not results:
                LOG.warning("No results from minibatch; skipping iteration")
                continue

            # 6. Run correctness for each result
            _run_correctness(results, cfg)

            # 7. Compute token lengths and composite score
            all_scored = []
            for r in results:
                answer = r.get("llm_answer", "")
                tok_len = _count_tokens(answer, tokenizer)
                all_scored.append((r["example_id"], answer, tok_len))
            mean_tok_len = sum(s[2] for s in all_scored) / len(all_scored)
            num_correct = sum(1 for r in results if r.get("is_correct", False))
            correctness_ratio = num_correct / len(results)
            shortness_score = 1.0 / (1.0 + mean_tok_len / cfg.shortness_scale)
            composite_score = 0.4 * shortness_score + 0.6 * correctness_ratio

            LOG.info(
                "Iteration %d: mean_tok_len=%.1f, correctness_ratio=%.2f (%d/%d), "
                "shortness_score=%.3f, composite_score=%.3f",
                iteration,
                mean_tok_len,
                correctness_ratio,
                num_correct,
                len(results),
                shortness_score,
                composite_score,
            )

            responses_with_correctness = [
                (
                    r["example_id"],
                    r.get("llm_answer", ""),
                    r.get("is_correct", False),
                    r.get("correctness_reasoning", ""),
                )
                for r in results
            ]

            # 8. Build reflector message and log it
            messages = build_reflector_message(
                cluster_descriptions=cluster_descriptions,
                current_deltas=current_deltas,
                cot_summary=cot_summary,
                responses_with_correctness=responses_with_correctness,
            )
            log_reflector_message(messages, iteration, cfg.output_dir)

            # 9. Call reflector
            reflector_output = call_reflector(cfg, messages)

            # 10. Validate and update deltas
            new_deltas = validate_reflector_output(
                reflector_output, cfg.expected_cluster_ids
            )
            summary_update = reflector_output.get("summary", "")
            LOG.info("Reflector summary: %s", summary_update)

            current_deltas = new_deltas
            if cot_summary:
                cot_summary += f"\n\n[Iteration {iteration}] {summary_update}"
            else:
                cot_summary = f"[Iteration {iteration}] {summary_update}"

            # Persist evolving summary so resume can restore it
            (out_dir / "cot_summary.txt").write_text(cot_summary)

            # 11. Track best (by composite score)
            if best_composite_score is None or composite_score > best_composite_score:
                best_composite_score = composite_score
                best_deltas = dict(current_deltas)
                _persist_best(cfg, best_deltas)
                LOG.info(
                    "New best at iteration %d: composite_score=%.3f",
                    iteration,
                    best_composite_score,
                )

            # 12. Update history and graph
            history.append({
                "iteration": iteration,
                "mean_token_length": mean_tok_len,
                "correctness_ratio": correctness_ratio,
                "shortness_score": shortness_score,
                "composite_score": composite_score,
            })
            update_graph(history, cfg.output_dir)

            # Save history to JSON for reproducibility
            _save_json(history, str(out_dir / "history.json"))

            if interrupted:
                LOG.info("Interrupted after iteration %d", iteration)
                break

    finally:
        # ---- Cleanup ----
        signal.signal(signal.SIGINT, original_sigint)

        if server is not None:
            try:
                server.stop()
            except Exception:
                LOG.exception("Error stopping SGLang during cleanup")

        # Ensure best is persisted
        if best_deltas is not None:
            _persist_best(cfg, best_deltas)
            LOG.info(
                "Evolution complete. Best composite score: %.3f",
                best_composite_score or 0,
            )
        else:
            LOG.warning("No best deltas recorded (no successful iteration)")

        LOG.info("Evolution outputs in %s", cfg.output_dir)
