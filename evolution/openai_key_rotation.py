"""
OpenAI API key rotation for evolution: load keys from a file at start;
on API errors, cycle to the next key and retry.  Exhausting all keys in
a single request raises; the *next* request starts from wherever the
cycle left off (so keys that recovered from rate-limits get retried).

Thread-safe: all state is protected by a lock.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List

LOG = logging.getLogger(__name__)

_lock = threading.Lock()
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
    with _lock:
        _keys = raw
        _index = 0
        _env_var = env_var
        os.environ[_env_var] = _keys[_index]
    LOG.info("Loaded %d OpenAI API key(s)", len(raw))


def num_keys() -> int:
    """Return the number of loaded keys."""
    with _lock:
        return len(_keys)


def get_key() -> str:
    """Return the current API key (thread-safe snapshot)."""
    with _lock:
        return _keys[_index]


def rotate_key() -> str:
    """Advance to the next key (cyclic wrap-around) and return it."""
    global _index
    with _lock:
        prev = _index
        _index = (_index + 1) % len(_keys)
        os.environ[_env_var] = _keys[_index]
        LOG.info("Rotated API key %d → %d (of %d)", prev + 1, _index + 1, len(_keys))
        return _keys[_index]


def is_openai_api_error(ex: BaseException) -> bool:
    """Return True for exceptions that should trigger key rotation (OpenAI API errors)."""
    mod = type(ex).__module__
    if mod.startswith("openai"):
        return True
    try:
        from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError
        if isinstance(ex, (APIError, APIConnectionError, RateLimitError, APITimeoutError)):
            return True
    except ImportError:
        pass
    return False


def is_initialized() -> bool:
    """True if init_keys has been called successfully (keys list non-empty)."""
    with _lock:
        return len(_keys) > 0
