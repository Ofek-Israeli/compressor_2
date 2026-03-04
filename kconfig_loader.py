"""
Kconfig loader for compressor_2 evolution.

Parses .config files (kconfiglib format) into EvolutionConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .evolution.config import EvolutionConfig


def _parse_config_file(path: str) -> Dict[str, object]:
    """Parse a kconfiglib .config into {KEY: value} (without CONFIG_ prefix)."""
    values: Dict[str, object] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            if key.startswith("CONFIG_"):
                key = key[len("CONFIG_"):]
            raw = raw.strip()
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1]
            if raw == "y":
                values[key] = True
            elif raw == "n":
                values[key] = False
            else:
                try:
                    values[key] = int(raw)
                except ValueError:
                    try:
                        values[key] = float(raw)
                    except ValueError:
                        values[key] = raw
    return values


def _parse_pool_indices(raw: str) -> Optional[List[int]]:
    """Parse 'all' or comma-separated ints."""
    raw = raw.strip()
    if not raw or raw.lower() == "all":
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _resolve_path(config_dir: Path, raw: str) -> str:
    """If raw is a relative path, resolve it relative to config_dir; else return as-is."""
    if not raw:
        return raw
    path = Path(raw)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return str(path)


def _validate_ga_options(cfg: EvolutionConfig) -> None:
    """Validate all GA-specific options; raise if any sentinel remains."""
    errors: List[str] = []

    if cfg.lambda_shortness < 0:
        errors.append("CONFIG_LAMBDA_SHORTNESS is required (got sentinel)")
    if cfg.lambda_correctness < 0:
        errors.append("CONFIG_LAMBDA_CORRECTNESS is required (got sentinel)")
    if cfg.population_size < 1:
        errors.append("CONFIG_POPULATION_SIZE must be >= 1 (got sentinel)")
    if cfg.ngen < 1:
        errors.append("CONFIG_NGEN must be >= 1 (got sentinel)")
    if cfg.cxpb < 0:
        errors.append("CONFIG_CXPB is required (got sentinel)")
    if cfg.mutpb < 0:
        errors.append("CONFIG_MUTPB is required (got sentinel)")
    if cfg.elitism_size < 0:
        errors.append("CONFIG_ELITISM_SIZE is required (got sentinel)")
    if cfg.selection not in ("tournament", "truncation"):
        errors.append(
            f"CONFIG_SELECTION must be 'tournament' or 'truncation' (got {cfg.selection!r})"
        )

    if cfg.selection == "tournament":
        if cfg.tournament_size < 2:
            errors.append("CONFIG_TOURNAMENT_SIZE must be >= 2 for tournament selection")
        elif cfg.tournament_size > cfg.population_size:
            errors.append(
                f"CONFIG_TOURNAMENT_SIZE ({cfg.tournament_size}) must be "
                f"<= CONFIG_POPULATION_SIZE ({cfg.population_size})"
            )
    elif cfg.selection == "truncation":
        if cfg.truncation_top_k < 2:
            errors.append("CONFIG_TRUNCATION_TOP_K must be >= 2 for truncation selection")

    if cfg.elitism_size >= cfg.population_size:
        errors.append(
            f"CONFIG_ELITISM_SIZE ({cfg.elitism_size}) must be "
            f"< CONFIG_POPULATION_SIZE ({cfg.population_size})"
        )

    if errors:
        raise ValueError(
            "GA config validation failed:\n  " + "\n  ".join(errors)
        )


def _safe_float(v: Dict[str, object], key: str, default: float) -> float:
    raw = v.get(key, default)
    if isinstance(raw, str):
        if raw == "":
            return default
        return float(raw)
    return float(raw)


def _safe_bool(v: Dict[str, object], key: str, default: bool) -> bool:
    raw = v.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("y", "yes", "true", "1")
    return bool(raw)


_VALID_METHODS = frozenset({
    "deap", "grid_search", "random_search", "spsa", "random_direction_2pt",
    "differential_evolution", "cmaes", "optuna_tpe", "smac", "tr_dfo",
    "skopt", "hybrid", "coordinate_then_random_direction",
})

_OPT_METHOD_MAP = {
    "OPT_METHOD_DEAP": "deap",
    "OPT_METHOD_GRID_SEARCH": "grid_search",
    "OPT_METHOD_RANDOM_SEARCH": "random_search",
    "OPT_METHOD_SPSA": "spsa",
    "OPT_METHOD_RANDOM_DIRECTION_2PT": "random_direction_2pt",
    "OPT_METHOD_DIFFERENTIAL_EVOLUTION": "differential_evolution",
    "OPT_METHOD_CMAES": "cmaes",
    "OPT_METHOD_OPTUNA_TPE": "optuna_tpe",
    "OPT_METHOD_SMAC": "smac",
    "OPT_METHOD_TR_DFO": "tr_dfo",
    "OPT_METHOD_SKOPT": "skopt",
    "OPT_METHOD_HYBRID": "hybrid",
    "OPT_METHOD_COORDINATE_THEN_RANDOM_DIRECTION": "coordinate_then_random_direction",
}


def _detect_method(v: Dict[str, object]) -> str:
    """Detect optimization method from OPT_METHOD_* bools (new Kconfig choice)
    with fallback to legacy OPTIMIZATION_METHOD string."""
    for key, method in _OPT_METHOD_MAP.items():
        if v.get(key) is True:
            return method
    return str(v.get("OPTIMIZATION_METHOD", "deap")).strip()


def _validate_zero_order_options(cfg: EvolutionConfig) -> None:
    """Validate zero-order-specific options (called when method != deap)."""
    errors: List[str] = []

    if cfg.optimization_method not in _VALID_METHODS:
        errors.append(
            f"CONFIG_OPTIMIZATION_METHOD must be one of {sorted(_VALID_METHODS)}, "
            f"got {cfg.optimization_method!r}"
        )

    if cfg.zero_order_max_evals < 1:
        errors.append("CONFIG_ZERO_ORDER_MAX_EVALS must be >= 1")
    if cfg.eval_minibatch_size < 1:
        errors.append("CONFIG_EVAL_MINIBATCH_SIZE must be >= 1")
    if cfg.deltas_bound_low >= cfg.deltas_bound_high:
        errors.append("CONFIG_DELTAS_BOUND_LOW must be < CONFIG_DELTAS_BOUND_HIGH")
    if cfg.llm_max_tokens < 1:
        errors.append("CONFIG_LLM_MAX_TOKENS must be >= 1")
    if cfg.lambda_shortness < 0:
        errors.append("CONFIG_LAMBDA_SHORTNESS is required for zero-order (fitness weights)")
    if cfg.lambda_correctness < 0:
        errors.append("CONFIG_LAMBDA_CORRECTNESS is required for zero-order (fitness weights)")

    if cfg.optimization_method in ("spsa", "random_direction_2pt"):
        if cfg.zero_order_max_evals < 2:
            errors.append(
                f"{cfg.optimization_method} requires ZERO_ORDER_MAX_EVALS >= 2 "
                f"(2 evals per step)"
            )

    if cfg.optimization_method in ("skopt", "grid_search"):
        if not cfg.eval_deterministic:
            errors.append(
                f"{cfg.optimization_method} requires CONFIG_EVAL_DETERMINISTIC=y"
            )

    if cfg.optimization_method == "grid_search":
        if cfg.grid_step <= 0:
            errors.append("CONFIG_GRID_STEP must be > 0")
        if cfg.grid_low >= cfg.grid_high:
            errors.append("CONFIG_GRID_LOW must be < CONFIG_GRID_HIGH")
        if cfg.grid_batch_size < 1:
            errors.append("CONFIG_GRID_BATCH_SIZE must be >= 1")

    if cfg.optimization_method == "coordinate_then_random_direction":
        # Phase 1 (coordinate)
        if cfg.coord_rd_coord_k < 1:
            errors.append("COORD_RD_COORD_K must be > 0")
        if cfg.coord_rd_coord_alpha0 <= 0:
            errors.append("COORD_RD_COORD_ALPHA0 must be > 0")
        if cfg.coord_rd_coord_alpha_min <= 0:
            errors.append("COORD_RD_COORD_ALPHA_MIN must be > 0")
        if cfg.coord_rd_coord_alpha_min > cfg.coord_rd_coord_alpha0:
            errors.append("COORD_RD_COORD_ALPHA_MIN must be <= COORD_RD_COORD_ALPHA0")
        if not (0 < cfg.coord_rd_coord_shrink < 1):
            errors.append("COORD_RD_COORD_SHRINK must be in (0, 1)")
        if cfg.coord_rd_coord_improvement_eps < 0:
            errors.append("COORD_RD_COORD_IMPROVEMENT_EPS must be >= 0")
        if cfg.coord_rd_coord_max_coords_total < 0:
            errors.append("COORD_RD_COORD_MAX_COORDS_TOTAL must be >= 0")
        # Phase 2 (random)
        if cfg.coord_rd_rand_alpha0 <= 0:
            errors.append("COORD_RD_RAND_ALPHA0 must be > 0")
        if cfg.coord_rd_rand_alpha_min <= 0:
            errors.append("COORD_RD_RAND_ALPHA_MIN must be > 0")
        if cfg.coord_rd_rand_alpha_min > cfg.coord_rd_rand_alpha0:
            errors.append("COORD_RD_RAND_ALPHA_MIN must be <= COORD_RD_RAND_ALPHA0")
        if not (0 < cfg.coord_rd_rand_shrink < 1):
            errors.append("COORD_RD_RAND_SHRINK must be in (0, 1)")
        if cfg.coord_rd_rand_improvement_eps < 0:
            errors.append("COORD_RD_RAND_IMPROVEMENT_EPS must be >= 0")
        if cfg.coord_rd_rand_dirs_per_iter < 1:
            errors.append("COORD_RD_RAND_DIRS_PER_ITER must be >= 1")
        if cfg.coord_rd_rand_dir_dist not in ("gaussian_unit", "rademacher_unit"):
            errors.append(
                f"COORD_RD_RAND_DIR_DIST must be 'gaussian_unit' or 'rademacher_unit', "
                f"got {cfg.coord_rd_rand_dir_dist!r}"
            )
        if cfg.coord_rd_rand_max_dir_pairs_total < 0:
            errors.append("COORD_RD_RAND_MAX_DIR_PAIRS_TOTAL must be >= 0")

    if errors:
        raise ValueError(
            "Zero-order config validation failed:\n  " + "\n  ".join(errors)
        )


def load_config(config_path: str, validate_ga: bool = True) -> EvolutionConfig:
    """Load a .config file and return an EvolutionConfig.

    All path options (paths to files or directories) are resolved relative to
    the directory containing the config file when they are relative. This
    allows the same .config to work regardless of current working directory.

    When OPTIMIZATION_METHOD != "deap", GA options are not validated even
    if validate_ga=True; zero-order options are validated instead.
    When validate_ga is False and method is deap, GA options are also
    skipped (for generate-evolution-processors).
    """
    p = Path(config_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config_dir = p.parent
    v = _parse_config_file(str(p))

    method = _detect_method(v)

    cfg = EvolutionConfig(
        financebench_jsonl=_resolve_path(config_dir, str(v.get("FINANCEBENCH_JSONL", ""))),
        pool_indices=_parse_pool_indices(str(v.get("FINANCEBENCH_POOL_INDICES", "all"))),
        cluster_descriptions_path=_resolve_path(config_dir, str(v.get("CLUSTER_DESCRIPTIONS_PATH", ""))),
        initial_delta_value=_safe_float(v, "INITIAL_DELTA_VALUE", 0.0),
        kmeans_path=_resolve_path(config_dir, str(v.get("KMEANS_PATH", ""))),
        embeddings_path=_resolve_path(config_dir, str(v.get("EMBEDDINGS_PATH", ""))),
        output_dir=_resolve_path(config_dir, str(v.get("OUTPUT_DIR", "outputs/evolution"))),
        minibatch_size=int(v.get("MINIBATCH_SIZE", 5)),
        max_iterations=int(v.get("MAX_ITERATIONS", 20)),
        correctness_model=str(v.get("CORRECTNESS_MODEL", "gpt-4o")),
        correctness_tolerance=float(v.get("CORRECTNESS_TOLERANCE", 0.10)),
        minions_repo=_resolve_path(config_dir, str(v.get("MINIONS_REPO", "/workspace/minions_channel"))),
        sglang_model_path=str(v.get("SGLANG_MODEL_PATH", "")),
        sglang_port=int(v.get("SGLANG_PORT", 8000)),
        sglang_extra_args=str(v.get("SGLANG_EXTRA_ARGS", "")),
        sglang_health_timeout=int(v.get("SGLANG_HEALTH_TIMEOUT", 300)),
        sglang_health_interval=int(v.get("SGLANG_HEALTH_INTERVAL", 5)),
        runner_config_path=_resolve_path(config_dir, str(v.get("RUNNER_CONFIG_PATH", ""))),
        runner_timeout_s=float(v.get("RUNNER_TIMEOUT_S", 300)),
        reflector_model=str(v.get("REFLECTOR_MODEL", "gpt-4o")),
        openai_api_key_env=str(v.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY")),
        openai_keys_file=_resolve_path(config_dir, str(v.get("OPENAI_KEYS_FILE", "/workspace/compressor_2/openai_keys.txt"))),
        reflector_temperature=_safe_float(v, "REFLECTOR_TEMPERATURE", 0.7),
        embedding_model=str(v.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")),
        gen_batch_size=int(v.get("GEN_BATCH_SIZE", 512)),
        # GA options
        lambda_shortness=_safe_float(v, "LAMBDA_SHORTNESS", -1.0),
        lambda_correctness=_safe_float(v, "LAMBDA_CORRECTNESS", -1.0),
        population_size=int(v.get("POPULATION_SIZE", -1)),
        ngen=int(v.get("NGEN", -1)),
        cxpb=_safe_float(v, "CXPB", -1.0),
        mutpb=_safe_float(v, "MUTPB", -1.0),
        elitism_size=int(v.get("ELITISM_SIZE", -1)),
        selection=str(v.get("SELECTION", "")),
        tournament_size=int(v.get("TOURNAMENT_SIZE", -1)),
        truncation_top_k=int(v.get("TRUNCATION_TOP_K", -1)),
        # Optimization method selector
        optimization_method=method,
        # Zero-order options
        zero_order_max_evals=int(v.get("ZERO_ORDER_MAX_EVALS", 100)),
        eval_minibatch_size=int(v.get("EVAL_MINIBATCH_SIZE", 5)),
        eval_seed=int(v.get("EVAL_SEED", 42)),
        eval_timeout_s=int(v.get("EVAL_TIMEOUT_S", 600)),
        eval_max_retries=int(v.get("EVAL_MAX_RETRIES", 1)),
        eval_failure_fitness=_safe_float(v, "EVAL_FAILURE_FITNESS", -1e9),
        deltas_bound_low=_safe_float(v, "DELTAS_BOUND_LOW", -2.0),
        deltas_bound_high=_safe_float(v, "DELTAS_BOUND_HIGH", 2.0),
        optimizer_seed=int(v.get("OPTIMIZER_SEED", 0)),
        eval_deterministic=_safe_bool(v, "EVAL_DETERMINISTIC", True),
        inference_seed=int(v.get("INFERENCE_SEED", 0)),
        llm_max_tokens=int(v.get("LLM_MAX_TOKENS", 2048)),
        enable_cache=_safe_bool(v, "ENABLE_CACHE", True),
        cache_round_decimals=int(v.get("CACHE_ROUND_DECIMALS", 6)),
        run_final_full_pool_eval=_safe_bool(v, "RUN_FINAL_FULL_POOL_EVAL", True),
        # Grid search
        grid_low=_safe_float(v, "GRID_LOW", -2.0),
        grid_high=_safe_float(v, "GRID_HIGH", 2.0),
        grid_step=_safe_float(v, "GRID_STEP", 0.1),
        grid_max_combos=int(v.get("GRID_MAX_COMBOS", 20000)),
        grid_allow_truncation=_safe_bool(v, "GRID_ALLOW_TRUNCATION", False),
        grid_batch_size=int(v.get("GRID_BATCH_SIZE", 64)),
        # Method-specific
        tr_dfo_method=str(v.get("TR_DFO_METHOD", "bobyqa")),
        skopt_n_random_starts=int(v.get("SKOPT_N_RANDOM_STARTS", 10)),
        optuna_n_trials=int(v.get("OPTUNA_N_TRIALS", 0)),
        zo_step_size=_safe_float(v, "ZO_STEP_SIZE", 0.01),
        zo_perturb_scale=_safe_float(v, "ZO_PERTURB_SCALE", 0.01),
        zo_num_directions=int(v.get("ZO_NUM_DIRECTIONS", 1)),
        zo_t=_safe_float(v, "ZO_T", 1.0),
        # Hybrid
        hybrid_global_method=str(v.get("HYBRID_GLOBAL_METHOD", "differential_evolution")),
        hybrid_global_evals=int(v.get("HYBRID_GLOBAL_EVALS", 0)),
        hybrid_local_evals=int(v.get("HYBRID_LOCAL_EVALS", 0)),
        # Coordinate-then-random-direction: Phase 1
        coord_rd_coord_k=int(v.get("COORD_RD_COORD_K", 10)),
        coord_rd_coord_alpha0=_safe_float(v, "COORD_RD_COORD_ALPHA0", 0.1),
        coord_rd_coord_alpha_min=_safe_float(v, "COORD_RD_COORD_ALPHA_MIN", 1e-6),
        coord_rd_coord_shrink=_safe_float(v, "COORD_RD_COORD_SHRINK", 0.5),
        coord_rd_coord_num_coords_per_iter=int(v.get("COORD_RD_COORD_NUM_COORDS_PER_ITER", 0)),
        coord_rd_coord_opportunistic=_safe_bool(v, "COORD_RD_COORD_OPPORTUNISTIC", False),
        coord_rd_coord_improvement_eps=_safe_float(v, "COORD_RD_COORD_IMPROVEMENT_EPS", 0.0),
        coord_rd_coord_sample_with_replacement=_safe_bool(v, "COORD_RD_COORD_SAMPLE_WITH_REPLACEMENT", False),
        coord_rd_coord_shuffle_each_iter=_safe_bool(v, "COORD_RD_COORD_SHUFFLE_EACH_ITER", True),
        coord_rd_coord_max_coords_total=int(v.get("COORD_RD_COORD_MAX_COORDS_TOTAL", 0)),
        # Coordinate-then-random-direction: Phase 2
        coord_rd_rand_alpha0=_safe_float(v, "COORD_RD_RAND_ALPHA0", 0.1),
        coord_rd_rand_alpha_min=_safe_float(v, "COORD_RD_RAND_ALPHA_MIN", 1e-6),
        coord_rd_rand_shrink=_safe_float(v, "COORD_RD_RAND_SHRINK", 0.5),
        coord_rd_rand_dir_dist=str(v.get("COORD_RD_RAND_DIR_DIST", "gaussian_unit")),
        coord_rd_rand_dirs_per_iter=int(v.get("COORD_RD_RAND_DIRS_PER_ITER", 1)),
        coord_rd_rand_improvement_eps=_safe_float(v, "COORD_RD_RAND_IMPROVEMENT_EPS", 0.0),
        coord_rd_rand_use_current_x_for_next_dir=_safe_bool(v, "COORD_RD_RAND_USE_CURRENT_X_FOR_NEXT_DIR", True),
        coord_rd_rand_max_dir_pairs_total=int(v.get("COORD_RD_RAND_MAX_DIR_PAIRS_TOTAL", 0)),
    )

    if method == "deap":
        if validate_ga:
            _validate_ga_options(cfg)
    else:
        _validate_zero_order_options(cfg)

    return cfg
