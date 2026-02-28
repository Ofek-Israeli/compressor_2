"""
GA Reflector: mutation and scratchpad-merge calls for the DEAP evolution loop.

Mutation: reflector proposes a full new delta vector (all clusters) + scratchpad.
Merge: given two parent scratchpads + fitness values, reflector produces a merged scratchpad.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .config import EvolutionConfig

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutation reflector
# ---------------------------------------------------------------------------

MUTATION_SYSTEM_PROMPT = (
    "You are a reflector for a steering system that adjusts LLM outputs by cluster. "
    "Your task is to propose a complete new delta vector (one value per cluster) to "
    "reduce verbosity without harming correctness. You receive: (1) cluster descriptions, "
    "(2) the current delta vector and scratchpad for this individual, (3) minibatch "
    "responses with per-example correctness and explanation. You MUST return a new delta "
    "vector that differs from the current one in at least one cluster. Negative deltas "
    "discourage tokens in that cluster; positive deltas encourage them. "
    'Output only a JSON object with two fields: "deltas" (object mapping cluster id '
    'string to number), "scratchpad" (string \u2014 your updated scratchpad for this '
    "evolution path, appending what you learnt this step). No other commentary."
)

MUTATION_USER_TEMPLATE = """\
## Cluster descriptions
{cluster_descriptions_json}

## Current delta vector
{current_deltas_json}

## Current scratchpad
{current_scratchpad}

## Minibatch responses (correctness and explanation per example)
{all_responses_with_correctness}

---
Propose a complete new delta vector (all clusters) to improve shortness without harming \
correctness. At least one cluster delta MUST differ from the current value. Update the \
scratchpad with what you learnt. \
IMPORTANT: Internally go cluster-by-cluster and decide keep/increase/decrease for each delta. 
Output only: {{"deltas": {{"<cluster_id>": <float>, ...}}, "scratchpad": "<string>"}}.\
"""

MUTATION_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mutation_output",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "deltas": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "scratchpad": {"type": "string"},
            },
            "required": ["deltas", "scratchpad"],
            "additionalProperties": False,
        },
    },
}


def _format_responses(
    results: List[Dict[str, Any]],
) -> str:
    """Format training results for the reflector prompt."""
    parts = []
    for i, r in enumerate(results, 1):
        eid = r.get("example_id", f"ex_{i}")
        answer = r.get("llm_answer", "")
        is_correct = r.get("is_correct", False)
        reasoning = r.get("correctness_reasoning", "")
        correct_label = "yes" if is_correct else "no"
        parts.append(
            f"### Response {i} (example: {eid})\n"
            f"Correct: {correct_label}\n"
            f"Explanation: {reasoning}\n\n"
            f"{answer}"
        )
    return "\n\n".join(parts)


def call_mutation_reflector(
    cfg: EvolutionConfig,
    cluster_descriptions: Dict[str, Any],
    current_deltas: Dict[str, float],
    scratchpad: str,
    training_results: List[Dict[str, Any]],
    *,
    output_dir: Optional[str] = None,
    generation: int = 0,
    index: int = 0,
) -> Optional[Tuple[Dict[str, float], str]]:
    """Call the reflector for mutation. Returns (new_deltas_dict, new_scratchpad) or None on failure."""
    api_key = os.environ.get(cfg.openai_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {cfg.openai_api_key_env} is not set")

    user_content = MUTATION_USER_TEMPLATE.format(
        cluster_descriptions_json=json.dumps(cluster_descriptions, indent=2),
        current_deltas_json=json.dumps(current_deltas, indent=2),
        current_scratchpad=scratchpad or "(empty \u2014 first mutation on this lineage)",
        all_responses_with_correctness=_format_responses(training_results),
    )
    messages = [
        {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if output_dir is not None:
        log_reflector_call(messages, "mutation", generation, index, output_dir)

    client = OpenAI(api_key=api_key)
    expected_keys = set(current_deltas.keys())

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=cfg.reflector_model,
                messages=messages,
                temperature=cfg.reflector_temperature,
                response_format=MUTATION_RESPONSE_SCHEMA,
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            new_deltas = data.get("deltas")
            new_scratchpad = data.get("scratchpad", "")

            if not isinstance(new_deltas, dict):
                LOG.warning("Mutation attempt %d: deltas not a dict (%r)", attempt + 1, type(new_deltas))
                continue

            parsed: Dict[str, float] = {}
            bad = False
            for k in expected_keys:
                v = new_deltas.get(k)
                if v is None:
                    LOG.warning("Mutation attempt %d: missing cluster %s", attempt + 1, k)
                    bad = True
                    break
                if not isinstance(v, (int, float)):
                    LOG.warning("Mutation attempt %d: cluster %s not numeric (%r)", attempt + 1, k, v)
                    bad = True
                    break
                parsed[k] = float(v)
            if bad:
                continue

            changed = any(
                abs(parsed[k] - current_deltas.get(k, 0.0)) > 1e-9 for k in expected_keys
            )
            if not changed:
                LOG.warning("Mutation attempt %d: all deltas unchanged", attempt + 1)
                continue

            return parsed, str(new_scratchpad)

        except Exception:
            LOG.exception("Mutation reflector call failed (attempt %d)", attempt + 1)

    LOG.warning("Mutation reflector failed after 2 attempts; skipping mutation")
    return None


# ---------------------------------------------------------------------------
# Merge reflector (crossover scratchpad merge)
# ---------------------------------------------------------------------------

MERGE_SYSTEM_PROMPT = (
    "You are a reflector that merges two scratchpads from two parent individuals "
    "in an evolutionary run. You receive two scratchpads and the fitness value of each "
    "parent. Your task is to produce a single merged scratchpad for their child. The "
    "merged scratchpad should combine the insights from both parents. If there are "
    "conflicting claims between the two scratchpads, prefer the claim from the parent "
    "with the higher fitness (more evidence-supported). Output only a JSON object with "
    'one field: "scratchpad" (string). No other commentary.'
)

MERGE_USER_TEMPLATE = """\
## Parent A (fitness: {fitness_a})
{scratchpad_a}

## Parent B (fitness: {fitness_b})
{scratchpad_b}

---
Merge these two scratchpads into one for the child. When claims conflict, favor the \
higher-fitness parent. Output only: {{"scratchpad": "<merged string>"}}.\
"""

MERGE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "merge_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scratchpad": {"type": "string"},
            },
            "required": ["scratchpad"],
            "additionalProperties": False,
        },
    },
}


def call_merge_reflector(
    cfg: EvolutionConfig,
    scratchpad_a: str,
    scratchpad_b: str,
    fitness_a: float,
    fitness_b: float,
    *,
    output_dir: Optional[str] = None,
    generation: int = 0,
    index: int = 0,
) -> str:
    """Call the reflector for scratchpad merge. Returns merged scratchpad string."""
    api_key = os.environ.get(cfg.openai_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {cfg.openai_api_key_env} is not set")

    user_content = MERGE_USER_TEMPLATE.format(
        fitness_a=f"{fitness_a:.4f}",
        scratchpad_a=scratchpad_a or "(empty)",
        fitness_b=f"{fitness_b:.4f}",
        scratchpad_b=scratchpad_b or "(empty)",
    )
    messages = [
        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if output_dir is not None:
        log_reflector_call(messages, "merge", generation, index, output_dir)

    client = OpenAI(api_key=api_key)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=cfg.reflector_model,
                messages=messages,
                temperature=cfg.reflector_temperature,
                response_format=MERGE_RESPONSE_SCHEMA,
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            merged = data.get("scratchpad")
            if isinstance(merged, str):
                return merged
            LOG.warning("Merge attempt %d: scratchpad not a string", attempt + 1)
        except Exception:
            LOG.exception("Merge reflector call failed (attempt %d)", attempt + 1)

    LOG.warning("Merge reflector failed; falling back to parent A scratchpad")
    return scratchpad_a


def log_reflector_call(
    messages: List[Dict[str, str]],
    call_type: str,
    generation: int,
    index: int,
    output_dir: str,
) -> None:
    """Write reflector messages to a log file."""
    out = Path(output_dir) / "reflector_prompts"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"gen_{generation:03d}_{call_type}_{index:03d}.txt"
    with open(log_path, "w") as f:
        for msg in messages:
            f.write(f"=== {msg['role'].upper()} ===\n")
            f.write(msg["content"])
            f.write("\n\n")
