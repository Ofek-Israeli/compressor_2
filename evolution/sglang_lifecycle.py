"""
SGLang server lifecycle: start, stop, and health-check.

Evolution runs on a 2-GPU pod.  SGLang is pinned to the **second** visible
GPU (cuda:1) via a CUDA_VISIBLE_DEVICES shim in :meth:`SGLangServer.start`
because SGLang's ``launch_server`` does not support a ``--device`` flag.
The shim restricts the child process to the single physical GPU whose index
is passed as *sglang_gpu_id* by the caller (computed from the parent's
CUDA_VISIBLE_DEVICES; see :func:`evolution.ga_driver._validate_2_gpus`).
See docs/2XGPU_pod_plan.md §3 / §7.2.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import time
from typing import Optional

import requests

from .config import EvolutionConfig

LOG = logging.getLogger(__name__)


class SGLangServer:
    """Manages an SGLang server subprocess pinned to a single GPU.

    The server is started **once** and kept running across all evaluations.
    Each evaluation's logit processor is sent per-request by the runner
    client (via ``custom_logit_processor`` in the JSON payload), so the
    server does not need to be restarted when switching processors.

    *sglang_gpu_id* is the **physical** GPU index string (e.g. ``"1"`` or
    ``"3"``) for the SGLang process.  It is used in a CUDA_VISIBLE_DEVICES
    shim because SGLang does not support a ``--device`` flag.
    """

    def __init__(
        self,
        cfg: EvolutionConfig,
        processor_path: Optional[str] = None,
        sglang_gpu_id: str = "1",
    ):
        self._cfg = cfg
        self._sglang_gpu_id = sglang_gpu_id
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._cfg.sglang_port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def start(self) -> None:
        """Start the SGLang server and block until healthy.

        SGLang is pinned to *sglang_gpu_id* via a CUDA_VISIBLE_DEVICES
        **shim**: the child env restricts visibility to that single physical
        GPU.  This is the recommended fallback from 2XGPU_pod_plan.md §7.2
        for servers that don't support a ``--device`` flag.
        """
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

        # Shim: SGLang has no --device flag, so restrict visibility to the
        # single physical GPU identified by _sglang_gpu_id.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self._sglang_gpu_id}
        LOG.info("Starting SGLang (GPU %s, shim CVD=%s): %s",
                 self._sglang_gpu_id, self._sglang_gpu_id, " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
