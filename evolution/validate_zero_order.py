"""
Zero-order validation: re-evaluate top-k training-fitness instances on a validation set.

Reads zero_order_history.jsonl, selects the k most successful (by f), runs each on
configurable validation indices via the same pipeline (generate-processor, runner,
correctness), produces a validation_fitness.png graph, and returns the path to the
best logit processor (.py file).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import EvolutionConfig
from .graph import VALIDATION_FITNESS_FILENAME, plot_validation_fitness
from .gpu_utils import validate_2_gpus, verify_gpu_pinning
from .objective import (
    deltas_list_to_dict,
    evaluate_x,
    generate_processor,
    save_json,
)
from .sglang_lifecycle import SGLangServer
from .validate_artifacts import build_initial_deltas

LOG = logging.getLogger(__name__)


def load_top_k_from_history(
    history_path: Path,
    k: int,
    indices_hash: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the top-k most successful records from zero_order_history.jsonl.

    Filters lines with status == "ok" and numeric f, sorts by f descending,
    returns at most k records. Each record has at least x, eval_id, f.

    If indices_hash is set, only lines with matching indices_hash are considered.
    """
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    successful: List[Dict[str, Any]] = []
    with open(history_path, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "ok":
                continue
            f = rec.get("f")
            if f is None or not isinstance(f, (int, float)):
                continue
            if indices_hash is not None and rec.get("indices_hash") != indices_hash:
                continue
            if "x" not in rec or not isinstance(rec["x"], list):
                continue
            successful.append(rec)

    if not successful:
        raise ValueError(
            f"No successful entries (status=ok with f) in {history_path}. "
            "Cannot run validation."
        )

    successful.sort(key=lambda r: (float(r["f"]), r.get("eval_id", 0)), reverse=True)
    return successful[:k]


def run_validation(
    history_path: Path,
    cfg: EvolutionConfig,
    validation_indices: List[int],
    top_k: int,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Run validation on top-k instances from history; return (graph_path, best_processor_path).

    Loads cluster_ids, starts SGLang, loads top-k from history. For each candidate:
    builds deltas from x, generates processor, evaluates on validation_indices.
    Plots validation fitness, selects best (max fitness; tie-break by larger eval_id).
    On per-candidate failure (e.g. SGLang/runner error), assigns eval_failure_fitness
    and continues.
    """
    if not validation_indices:
        raise ValueError(
            "validation_indices must be non-empty. Set CONFIG_VALIDATION_INDICES or --validation-indices."
        )

    # Correctness evaluation uses OpenAI; ensure keys are loaded
    if getattr(cfg, "openai_keys_file", ""):
        from .openai_key_rotation import init_keys
        init_keys(cfg.openai_keys_file, cfg.openai_api_key_env)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cluster IDs and dimensions
    _, cluster_ids = build_initial_deltas(cfg)
    cfg.expected_cluster_ids = cluster_ids

    # Load top-k from history (no indices_hash filter so any run works)
    candidates = load_top_k_from_history(history_path, top_k, indices_hash=None)
    LOG.info("Validating top %d candidates (by training fitness) on %d validation indices", len(candidates), len(validation_indices))

    # 2-GPU setup (same as optimize_driver)
    sglang_gpu_id = validate_2_gpus()
    server_holder: Dict[str, Any] = {"server": None}
    srv = SGLangServer(cfg, sglang_gpu_id=sglang_gpu_id)
    srv.start()
    server_holder["server"] = srv
    import time
    if getattr(srv, "_proc", None) is not None:
        time.sleep(2)
        verify_gpu_pinning(srv._proc.pid)
    LOG.info("SGLang started for validation")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.sglang_model_path)

    ranks: List[int] = []
    eval_ids: List[int] = []
    validation_fitnesses: List[float] = []
    processor_paths: List[Path] = []

    for rank, rec in enumerate(candidates, start=1):
        x = rec["x"]
        eval_id = rec.get("eval_id", rank - 1)
        ranks.append(rank)
        eval_ids.append(eval_id)

        deltas_dict = deltas_list_to_dict(
            [float(v) for v in x], cluster_ids
        )
        deltas_path = str(output_dir / f"_validation_deltas_eval{eval_id}.json")
        processor_path = output_dir / f"validation_processor_eval{eval_id}.py"
        save_json(deltas_dict, deltas_path)

        try:
            generate_processor(cfg, deltas_path, str(processor_path))
        except Exception as e:
            LOG.warning("generate_processor failed for eval_id=%s: %s", eval_id, e)
            validation_fitnesses.append(cfg.eval_failure_fitness)
            processor_paths.append(processor_path)
            continue

        try:
            f_val, _ = evaluate_x(
                [float(v) for v in x],
                cluster_ids,
                validation_indices,
                cfg,
                tokenizer,
                server_holder,
                output_dir,
                sglang_gpu_id,
                pre_generated_processor_path=str(processor_path),
            )
            validation_fitnesses.append(f_val)
        except Exception as e:
            LOG.warning("evaluate_x (validation) failed for eval_id=%s: %s", eval_id, e)
            validation_fitnesses.append(cfg.eval_failure_fitness)
        processor_paths.append(processor_path)

    # Plot: rank vs validation fitness
    graph_path = output_dir / VALIDATION_FITNESS_FILENAME
    plot_validation_fitness(
        ranks,
        validation_fitnesses,
        graph_path,
        title="Validation fitness (top-k from history)",
    )

    # Best: max validation fitness; tie-break by larger eval_id
    best_idx = 0
    for i in range(1, len(validation_fitnesses)):
        if validation_fitnesses[i] > validation_fitnesses[best_idx]:
            best_idx = i
        elif validation_fitnesses[i] == validation_fitnesses[best_idx] and eval_ids[i] > eval_ids[best_idx]:
            best_idx = i
    best_processor_path = processor_paths[best_idx]

    LOG.info(
        "Validation complete: best eval_id=%s validation_fitness=%.4f processor=%s",
        eval_ids[best_idx], validation_fitnesses[best_idx], best_processor_path,
    )
    return graph_path, best_processor_path
