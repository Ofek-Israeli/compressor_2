"""
OpenAI API key rotation for evolution: load keys from a file at start;
on API errors, switch to the next key and retry. Exhausting all keys raises.
"""

from __future__ import annotations

import os
from typing import List

# Module-level state
_keys: List[str] = []
_index: int = 0
_env_var: str = "OPENAI_API_KEY"


def init_keys(filepath: str, env_var: str) -> None:
    """
    Load API keys from file (one per line), set env to the first key.
    If the file is missing or has no keys, raise and do not start — run terminates.
    """
    global _keys, _index, _env_var
    if not filepath or not filepath.strip():
        raise FileNotFoundError(
            "OpenAI keys file path is empty; evolution requires a keys file. "
            "Set CONFIG_OPENAI_KEYS_FILE in Kconfig."
        )
    path = os.path.abspath(os.path.expanduser(filepath.strip()))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"OpenAI keys file not found: {path}. "
            "Evolution will not start; create the file or fix CONFIG_OPENAI_KEYS_FILE."
        )
    with open(path, "r") as f:
        raw = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not raw:
        raise ValueError(
            f"OpenAI keys file is empty or has no valid lines: {path}. "
            "Add at least one API key (one per line)."
        )
    _keys = raw
    _index = 0
    _env_var = env_var
    os.environ[_env_var] = _keys[_index]


def rotate_to_next_key() -> bool:
    """
    Switch to the next key in the list; update env. Return True if a next key
    was set, False if all keys have been tried.
    """
    global _index
    _index += 1
    if _index < len(_keys):
        os.environ[_env_var] = _keys[_index]
        return True
    return False


def is_openai_api_error(ex: BaseException) -> bool:
    """Return True for exceptions that should trigger key rotation (OpenAI API errors)."""
    mod = type(ex).__module__
    if mod.startswith("openai"):
        return True
    # Explicitly allow common OpenAI exception names from openai package
    try:
        from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError
        if isinstance(ex, (APIError, APIConnectionError, RateLimitError, APITimeoutError)):
            return True
    except ImportError:
        pass
    return False


def is_initialized() -> bool:
    """True if init_keys has been called successfully (keys list non-empty)."""
    return len(_keys) > 0
