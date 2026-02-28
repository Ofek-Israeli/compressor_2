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


def load_config(config_path: str, validate_ga: bool = True) -> EvolutionConfig:
    """Load a .config file and return an EvolutionConfig.

    All path options (paths to files or directories) are resolved relative to
    the directory containing the config file when they are relative. This
    allows the same .config to work regardless of current working directory.

    If validate_ga is False, GA options are not validated (for use by
    generate-evolution-processors when only path/embedding options are needed).
    """
    p = Path(config_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config_dir = p.parent
    v = _parse_config_file(str(p))

    cfg = EvolutionConfig(
        financebench_jsonl=_resolve_path(config_dir, str(v.get("FINANCEBENCH_JSONL", ""))),
        pool_indices=_parse_pool_indices(str(v.get("FINANCEBENCH_POOL_INDICES", "all"))),
        cluster_descriptions_path=_resolve_path(config_dir, str(v.get("CLUSTER_DESCRIPTIONS_PATH", ""))),
        initial_deltas_path=_resolve_path(config_dir, str(v.get("INITIAL_DELTAS_PATH", ""))),
        kmeans_path=_resolve_path(config_dir, str(v.get("KMEANS_PATH", ""))),
        embeddings_path=_resolve_path(config_dir, str(v.get("EMBEDDINGS_PATH", ""))),
        output_dir=_resolve_path(config_dir, str(v.get("OUTPUT_DIR", "outputs/evolution"))),
        minibatch_size=int(v.get("MINIBATCH_SIZE", 5)),
        max_iterations=int(v.get("MAX_ITERATIONS", 20)),
        shortness_scale=int(v.get("SHORTNESS_SCALE", 300)),
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
    )

    if validate_ga:
        _validate_ga_options(cfg)

    return cfg
