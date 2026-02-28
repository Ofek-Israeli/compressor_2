"""
Correctness evaluation using OpenAI API only (no minions dependency).

Same prompt and logic as learning_grammar/correctness.py for evolution use.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List

LOG = logging.getLogger(__name__)


@dataclass
class CorrectnessResult:
    """Result of a single correctness evaluation."""
    is_correct: bool
    confidence: float
    reasoning: str


def evaluate_one(
    predicted: str,
    ground_truth: List[str],
    question: str,
    model: str = "gpt-4o",
    tolerance: float = 0.10,
    api_key_env: str = "OPENAI_API_KEY",
) -> CorrectnessResult:
    """
    Evaluate one prediction against ground truth using OpenAI chat completion.
    Uses the same prompt as learning_grammar/correctness.py.
    """
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} is not set (required for correctness evaluation)"
        )
    client = OpenAI(api_key=api_key)

    gt_str = ground_truth[0] if ground_truth else "N/A"
    prompt = f"""You are evaluating whether a predicted answer is correct.

QUESTION: {question}

GROUND TRUTH: {gt_str}

PREDICTED: {predicted}

RULES:
1. For numerical answers: Allow {tolerance:.0%} tolerance
2. For yes/no: Must match exactly
3. For qualitative: Key facts must align
4. Ignore formatting differences

Respond with JSON:
{{"is_correct": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        text = response.choices[0].message.content or ""
        json_match = re.search(r'\{[^{}]*"is_correct"[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return CorrectnessResult(
                is_correct=data.get("is_correct", False),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "")),
            )
    except Exception as e:
        LOG.warning("Correctness evaluation failed: %s", e)

    return CorrectnessResult(
        is_correct=False,
        confidence=0.0,
        reasoning="Evaluation failed",
    )
