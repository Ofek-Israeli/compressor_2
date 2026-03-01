"""
Shared objective evaluator — single evaluation pipeline used by both GA and zero-order.

Zero-order is **not allowed** to re-implement generate-processor, SGLang
lifecycle, runner config, training set, or correctness; it must call through
this module only.  See docs/zero_order_opt_plan.md §3.1.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import EvolutionConfig
from .sglang_lifecycle import SGLangServer

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Delta list <-> dict conversion (canonical order)
# ---------------------------------------------------------------------------

def deltas_dict_to_list(d: Dict[str, float], cluster_ids: List[str]) -> List[float]:
    return [float(d[cid]) for cid in cluster_ids]


def deltas_list_to_dict(lst: List[float], cluster_ids: List[str]) -> Dict[str, float]:
    return {cid: lst[i] for i, cid in enumerate(cluster_ids)}


# ---------------------------------------------------------------------------
# Processor generation
# ---------------------------------------------------------------------------

def generate_processor(
    cfg: EvolutionConfig,
    deltas_path: str,
    output_path: str,
) -> None:
    """Run generate-processor subprocess on cuda:0 (embedding GPU).

    Inherits parent env as-is (no CUDA_VISIBLE_DEVICES override).
    The embedding model is pinned to cuda:0 via --device.
    See docs/2XGPU_pod_plan.md §3 / §7.1.
    """
    cmd = [
        sys.executable, "-m", "compressor_2", "generate-processor",
        cfg.kmeans_path, deltas_path,
        "-o", output_path,
        "--tokenizer", cfg.sglang_model_path,
        "--embedding-model", cfg.embedding_model,
        "--embeddings", cfg.embeddings_path,
        "-b", str(cfg.gen_batch_size),
        "--device", "cuda:0",
    ]
    LOG.info("Running generate-processor (device cuda:0) …")
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        LOG.error(
            "generate-processor failed after %.1fs (exit %s). stderr:\n%s",
            elapsed, result.returncode, result.stderr,
        )
        raise RuntimeError(f"generate-processor failed (exit {result.returncode})")
    LOG.info("generate-processor completed in %.1fs", elapsed)


# ---------------------------------------------------------------------------
# Runner config and training-set execution
# ---------------------------------------------------------------------------

def build_runner_config(
    base_config_path: str,
    indices: List[int],
    cfg: EvolutionConfig,
) -> str:
    """Create a temp YAML runner config for the given example indices."""
    import yaml

    with open(base_config_path, "r") as f:
        runner_cfg = yaml.safe_load(f)
    runner_cfg["example_indices"] = indices
    runner_cfg["concurrency"] = len(indices)
    runner_cfg["model_id"] = cfg.sglang_model_path
    if "sglang" not in runner_cfg or not isinstance(runner_cfg.get("sglang"), dict):
        runner_cfg["sglang"] = {}
    runner_cfg["sglang"]["base_url"] = f"http://localhost:{cfg.sglang_port}"
    runner_cfg["sglang"]["timeout_s"] = cfg.runner_timeout_s

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="ga_runner_",
    )
    yaml.dump(runner_cfg, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def run_training_set(
    runner_config_path: str,
    financebench_jsonl: str,
    processor_path: str,
) -> List[Dict[str, Any]]:
    import yaml

    with open(runner_config_path, "r") as f:
        runner_cfg = yaml.safe_load(f)
    n_examples = len(runner_cfg.get("example_indices", []))
    concurrency = runner_cfg.get("concurrency", n_examples)
    sglang_cfg = runner_cfg.get("sglang", {}) or {}
    runner_timeout_s = sglang_cfg.get("timeout_s", 300)
    LOG.info(
        "financebench_runner: n_examples=%s concurrency=%s timeout_s=%s processor=%s",
        n_examples, concurrency, runner_timeout_s, processor_path,
    )
    LOG.info(
        "Blocking on financebench_runner subprocess (runner logs follow); "
        "may take up to ~%ss per request.",
        runner_timeout_s,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_output = tmp.name
    cmd = [
        sys.executable, "-m", "financebench_runner",
        "--config", runner_config_path,
        "--input", financebench_jsonl,
        "--output", tmp_output,
        "--logit-processor", processor_path,
    ]
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        LOG.error("financebench_runner failed after %.1fs (exit %s)", elapsed, result.returncode)
        raise RuntimeError(f"financebench_runner failed (exit {result.returncode})")
    LOG.info("financebench_runner completed in %.1fs", elapsed)
    with open(tmp_output, "r") as f:
        results = json.load(f)
    os.unlink(tmp_output)
    return results


def run_correctness(results: List[Dict[str, Any]], cfg: EvolutionConfig) -> None:
    from .correctness_openai import evaluate_one

    def _eval_one(r: Dict[str, Any]) -> None:
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

    n = len(results)
    if n <= 1:
        for r in results:
            _eval_one(r)
        return

    with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
        futs = [pool.submit(_eval_one, r) for r in results]
        for fut in futs:
            fut.result()


# ---------------------------------------------------------------------------
# Core evaluation: x_list -> fitness
# ---------------------------------------------------------------------------

def evaluate_x(
    x_list: List[float],
    cluster_ids: List[str],
    evaluation_indices: List[int],
    cfg: EvolutionConfig,
    tokenizer: Any,
    server_holder: Dict[str, Any],
    out_dir: Path,
    sglang_gpu_id: str,
    pre_generated_processor_path: Optional[str] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Full evaluation pipeline: generate-processor -> training set -> correctness -> fitness.

    SGLang is kept running across evaluations (started once by the caller).
    Returns (fitness_scalar, training_results).
    """
    deltas_dict = deltas_list_to_dict(x_list, cluster_ids)

    if pre_generated_processor_path is not None:
        processor_path = pre_generated_processor_path
    else:
        deltas_path = str(out_dir / "_eval_deltas.json")
        save_json(deltas_dict, deltas_path)
        processor_path = str(out_dir / "_eval_processor.py")
        generate_processor(cfg, deltas_path, processor_path)

    srv: Optional[SGLangServer] = server_holder.get("server")
    if srv is None or not srv.is_running():
        LOG.warning("SGLang not running; restarting …")
        if srv is not None:
            try:
                srv.stop()
            except Exception:
                pass
        srv = SGLangServer(cfg, sglang_gpu_id=sglang_gpu_id)
        srv.start()
        server_holder["server"] = srv

    tmp_runner_cfg = build_runner_config(
        cfg.runner_config_path, evaluation_indices, cfg,
    )
    try:
        results = run_training_set(
            tmp_runner_cfg, cfg.financebench_jsonl, processor_path,
        )
    finally:
        os.unlink(tmp_runner_cfg)

    if not results:
        LOG.warning("No results from training set evaluation")
        return 0.0, []

    run_correctness(results, cfg)

    total_tok_len = 0
    num_correct = 0
    for r in results:
        answer = r.get("llm_answer", "")
        tok_len = len(tokenizer.encode(answer, add_special_tokens=False))
        r["tok_len"] = tok_len
        total_tok_len += tok_len
        if r.get("is_correct", False):
            num_correct += 1

    mean_tok_len = total_tok_len / len(results)
    correctness_ratio = num_correct / len(results)
    shortness_score = 1.0 / (1.0 + mean_tok_len / cfg.shortness_scale)
    fitness = cfg.lambda_shortness * shortness_score + cfg.lambda_correctness * correctness_ratio

    LOG.info(
        "  eval: mean_tok=%.0f  correct=%d/%d  shortness=%.3f  fitness=%.4f",
        mean_tok_len, num_correct, len(results), shortness_score, fitness,
    )
    return fitness, results


# ---------------------------------------------------------------------------
# Minimizer wrapper (negate fitness, clip bounds)
# ---------------------------------------------------------------------------

def make_objective_for_minimizer(
    cluster_ids: List[str],
    evaluation_indices: List[int],
    cfg: EvolutionConfig,
    tokenizer: Any,
    server_holder: Dict[str, Any],
    out_dir: Path,
    sglang_gpu_id: str,
    bounds_low: float,
    bounds_high: float,
) -> Callable[[List[float]], float]:
    """Return a callable that clips x into bounds, evaluates, and returns -f.

    Suitable for passing to minimization libraries (scipy, pdfo, etc.).
    """
    def _objective(x: List[float]) -> float:
        clipped = [max(bounds_low, min(bounds_high, xi)) for xi in x]
        f, _ = evaluate_x(
            clipped, cluster_ids, evaluation_indices, cfg,
            tokenizer, server_holder, out_dir, sglang_gpu_id,
        )
        return -f

    return _objective
