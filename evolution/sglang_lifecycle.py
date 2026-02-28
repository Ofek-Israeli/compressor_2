"""
SGLang server lifecycle: start, stop, and health-check.
"""

from __future__ import annotations

import logging
import shlex
import signal
import subprocess
import time
from typing import Optional

import requests

from .config import EvolutionConfig

LOG = logging.getLogger(__name__)


class SGLangServer:
    """Manages an SGLang server subprocess."""

    def __init__(self, cfg: EvolutionConfig, processor_path: str):
        self._cfg = cfg
        self._processor_path = processor_path
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._cfg.sglang_port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def start(self) -> None:
        """Start the SGLang server and block until healthy."""
        if self._proc is not None:
            LOG.warning("SGLang process already exists (pid %s); stopping first", self._proc.pid)
            self.stop()

        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self._cfg.sglang_model_path,
            "--port", str(self._cfg.sglang_port),
            "--enable-custom-logit-processor",
        ]
        if self._cfg.sglang_extra_args:
            cmd.extend(shlex.split(self._cfg.sglang_extra_args))

        LOG.info("Starting SGLang: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        LOG.info("SGLang started (pid %s); waiting for health...", self._proc.pid)
        self._wait_healthy()

    def stop(self) -> None:
        """Stop the SGLang server process."""
        if self._proc is None:
            return
        pid = self._proc.pid
        LOG.info("Stopping SGLang (pid %s)...", pid)
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            LOG.warning("SGLang did not exit after SIGTERM; sending SIGKILL")
            self._proc.kill()
            self._proc.wait(timeout=10)
        LOG.info("SGLang stopped (pid %s)", pid)
        self._proc = None

    def is_running(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def _wait_healthy(self) -> None:
        """Poll the health endpoint until success or timeout."""
        deadline = time.monotonic() + self._cfg.sglang_health_timeout
        interval = self._cfg.sglang_health_interval
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                self._log_stderr_on_exit()
                raise RuntimeError(
                    f"SGLang process exited with code {self._proc.returncode} during health wait"
                )
            try:
                r = requests.get(self.health_url, timeout=5)
                if r.status_code == 200:
                    LOG.info("SGLang healthy")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(interval)
        raise RuntimeError(
            f"SGLang did not become healthy within {self._cfg.sglang_health_timeout}s"
        )

    def _log_stderr_on_exit(self) -> None:
        """Read and log subprocess stderr when the process has exited (for debugging)."""
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            stderr_bytes = self._proc.stderr.read()
            if stderr_bytes:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                LOG.error("SGLang stderr:\n%s", stderr_text.rstrip())
        except Exception:
            LOG.exception("Failed to read SGLang stderr")
