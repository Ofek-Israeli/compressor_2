"""EvolutionConfig dataclass — all options for the evolution loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvolutionConfig:
    # --- Data paths ---
    financebench_jsonl: str = ""
    pool_indices: Optional[List[int]] = None  # None means "all"
    cluster_descriptions_path: str = ""
    initial_deltas_path: str = ""
    kmeans_path: str = ""
    embeddings_path: str = ""
    output_dir: str = "outputs/evolution"

    # --- Evolution parameters ---
    minibatch_size: int = 5
    max_iterations: int = 20
    shortness_scale: int = 300  # scale in shortness_score = 1 / (1 + mean_tok_len / scale)

    # --- SGLang server ---
    sglang_model_path: str = ""
    sglang_port: int = 8000
    sglang_extra_args: str = ""
    sglang_health_timeout: int = 300
    sglang_health_interval: int = 5

    # --- FinanceBench runner ---
    runner_config_path: str = ""
    runner_timeout_s: float = 300.0  # SGLang request timeout for minibatch runs

    # --- Correctness (learning_grammar) ---
    correctness_model: str = "gpt-4o"
    correctness_tolerance: float = 0.10
    minions_repo: str = "/workspace/minions_channel"

    # --- Reflector (OpenAI) ---
    reflector_model: str = "gpt-4o"
    openai_api_key_env: str = "OPENAI_API_KEY"
    reflector_temperature: float = 0.7

    # --- Generate processor ---
    embedding_model: str = "all-MiniLM-L6-v2"
    gen_batch_size: int = 512

    # --- Genetic Algorithm (DEAP) ---
    lambda_shortness: float = -1.0
    lambda_correctness: float = -1.0
    population_size: int = -1
    ngen: int = -1
    cxpb: float = -1.0
    mutpb: float = -1.0
    elitism_size: int = -1
    selection: str = ""
    tournament_size: int = -1
    truncation_top_k: int = -1

    # --- Optimization method selector ---
    optimization_method: str = "deap"

    # --- Zero-order: evaluation budget and failure ---
    zero_order_max_evals: int = 100
    eval_minibatch_size: int = 5
    eval_seed: int = 42
    eval_timeout_s: int = 600
    eval_max_retries: int = 1
    eval_failure_fitness: float = -1e9

    # --- Zero-order: bounds ---
    deltas_bound_low: float = -2.0
    deltas_bound_high: float = 2.0
    optimizer_seed: int = 0

    # --- Zero-order: determinism / LLM ---
    eval_deterministic: bool = True
    inference_seed: int = 0
    llm_max_tokens: int = 2048

    # --- Zero-order: cache ---
    enable_cache: bool = True
    cache_round_decimals: int = 6

    # --- Zero-order: final full-pool eval ---
    run_final_full_pool_eval: bool = True

    # --- Grid search ---
    grid_low: float = -2.0
    grid_high: float = 2.0
    grid_step: float = 0.1
    grid_max_combos: int = 20000
    grid_allow_truncation: bool = False
    grid_batch_size: int = 64

    # --- Method-specific ---
    tr_dfo_method: str = "bobyqa"
    skopt_n_random_starts: int = 10
    optuna_n_trials: int = 0
    zo_step_size: float = 0.01
    zo_perturb_scale: float = 0.01
    zo_num_directions: int = 1
    zo_t: float = 1.0

    # --- Hybrid ---
    hybrid_global_method: str = "differential_evolution"
    hybrid_global_evals: int = 0
    hybrid_local_evals: int = 0

    # --- Derived (set by loader/driver) ---
    expected_cluster_ids: List[str] = field(default_factory=list)
