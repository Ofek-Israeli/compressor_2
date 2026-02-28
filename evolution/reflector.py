"""
Reflector: build prompt, call OpenAI with JSON schema enforcement, validate output.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .config import EvolutionConfig

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a reflector for a steering system that adjusts LLM outputs by cluster. "
    "Your task is to spot padding, verbosity, and fluff in the given model responses "
    "and propose per-cluster steering deltas (a float per cluster id) so that future "
    "responses become shorter while maintaining or improving correctness. You will "
    "receive: (1) cluster descriptions (what each cluster id means), (2) the deltas "
    "that were used for this minibatch (the baseline you are adjusting from), (3) "
    "all minibatch model responses with per-example correctness and explanation, and "
    "(4) the evolving summary of what you have learnt so far (CoT). You must output a "
    'JSON object with two fields: "deltas" (object mapping each cluster id string "0", '
    '"1", ... to a float) and "summary" (your update to the evolving summary: 2–4 '
    "sentences on what you learnt this round). Negative deltas discourage tokens in "
    "that cluster; positive can boost. Do not harm correctness when reducing verbosity."
)

USER_TEMPLATE = """\
## Cluster descriptions
{cluster_descriptions_json}

## Deltas used for this minibatch (baseline to adjust from)
{current_deltas_json}

## Evolving summary (what you have learnt so far; you update this each round and it is served back to you in later iterations)
{cot_summary}

## All minibatch responses (with correctness and explanation per example)
{all_responses_with_correctness}

---
Reduce verbosity in these responses without harming correctness. Given the deltas that \
were used for this minibatch (above), propose updated deltas *for each* cluster so that \
responses become shorter while maintaining or improving correctness. Consider the \
correctness explanations when adjusting deltas. You must also update the evolving \
summary: provide 2–4 sentences on what you learnt this round. Output only a JSON \
object with two fields: "deltas" (object with string keys "0", "1", ... and float \
values) and "summary" (string, your update to the evolving summary). No other commentary. \ 
Try to think of suitable deltas to many clusters.
"""


def _build_json_schema(cluster_ids: List[str]) -> Dict[str, Any]:
    """Build OpenAI response_format JSON schema enforcing exact cluster id keys."""
    delta_properties = {cid: {"type": "number"} for cid in cluster_ids}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "reflector_output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "deltas": {
                        "type": "object",
                        "properties": delta_properties,
                        "required": cluster_ids,
                        "additionalProperties": False,
                    },
                    "summary": {"type": "string"},
                },
                "required": ["deltas", "summary"],
                "additionalProperties": False,
            },
        },
    }


def _format_responses_with_correctness(
    responses: List[Tuple[str, str, bool, str]],
) -> str:
    """Format (example_id, llm_answer, is_correct, reasoning) for the prompt."""
    parts = []
    for i, (eid, text, is_correct, reasoning) in enumerate(responses, 1):
        correct_label = "yes" if is_correct else "no"
        parts.append(
            f"### Response {i} (example: {eid})\n"
            f"Correct: {correct_label}\n"
            f"Explanation: {reasoning}\n\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def build_reflector_message(
    cluster_descriptions: Dict[str, Any],
    current_deltas: Dict[str, float],
    cot_summary: str,
    responses_with_correctness: List[Tuple[str, str, bool, str]],
) -> List[Dict[str, str]]:
    """Build the full messages list (system + user) for the reflector call.

    responses_with_correctness: list of (example_id, llm_answer, is_correct, reasoning).
    """
    user_content = USER_TEMPLATE.format(
        cluster_descriptions_json=json.dumps(cluster_descriptions, indent=2),
        current_deltas_json=json.dumps(current_deltas, indent=2),
        cot_summary=cot_summary or "First iteration; no prior learnings.",
        all_responses_with_correctness=_format_responses_with_correctness(
            responses_with_correctness
        ),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def log_reflector_message(
    messages: List[Dict[str, str]],
    iteration: int,
    output_dir: str,
) -> None:
    """Write the full reflector message to a log file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"reflector_message_{iteration:03d}.txt"
    with open(log_path, "w") as f:
        for msg in messages:
            f.write(f"=== {msg['role'].upper()} ===\n")
            f.write(msg["content"])
            f.write("\n\n")
    LOG.info("Logged reflector message to %s", log_path)


def call_reflector(
    cfg: EvolutionConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Call OpenAI with JSON schema enforcement; return parsed JSON dict."""
    api_key = os.environ.get(cfg.openai_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {cfg.openai_api_key_env} is not set"
        )
    client = OpenAI(api_key=api_key)
    schema = _build_json_schema(cfg.expected_cluster_ids)
    LOG.info("Calling reflector (%s)...", cfg.reflector_model)
    response = client.chat.completions.create(
        model=cfg.reflector_model,
        messages=messages,
        temperature=cfg.reflector_temperature,
        response_format=schema,
    )
    raw = response.choices[0].message.content
    LOG.debug("Reflector raw response: %s", raw[:200])
    return json.loads(raw)


def validate_reflector_output(
    output: Dict[str, Any],
    expected_ids: List[str],
) -> Dict[str, float]:
    """Validate reflector output and return the deltas dict.

    Raises ValueError if keys don't match or values aren't numeric.
    """
    if "deltas" not in output:
        raise ValueError("Reflector output missing 'deltas' key")
    if "summary" not in output:
        raise ValueError("Reflector output missing 'summary' key")

    deltas = output["deltas"]
    got = set(deltas.keys())
    want = set(expected_ids)
    if got != want:
        missing = want - got
        extra = got - want
        raise ValueError(
            f"Deltas key mismatch: missing={missing}, extra={extra}"
        )
    for cid, val in deltas.items():
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"Delta for cluster {cid!r} is not numeric: {val!r}"
            )
    return {cid: float(deltas[cid]) for cid in expected_ids}
