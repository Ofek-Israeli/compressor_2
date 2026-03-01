"""
Shared 2-GPU validation and pinning verification helpers.

Used by both ga_driver (DEAP) and optimize_driver (zero-order).
See docs/2XGPU_pod_plan.md for the full specification.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, Optional

LOG = logging.getLogger(__name__)


def validate_2_gpus() -> str:
    """Validate exactly 2 visible GPUs for the 2-GPU evolution policy.

    Returns the physical GPU ID string for the SGLang shim (the second
    visible GPU).  Embedding always uses ``cuda:0`` via device selection;
    SGLang needs the physical ID because it doesn't support a ``--device``
    flag (see docs/2XGPU_pod_plan.md §7.2 — shim approach).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        import torch
        n = torch.cuda.device_count()
        if n != 2:
            raise RuntimeError(
                f"This repo must run on a 2-GPU pod. Expected exactly 2 "
                f"visible GPUs; found {n}. If running on a larger host, set "
                f"CUDA_VISIBLE_DEVICES to exactly two GPUs."
            )
        return "1"

    tokens = [t.strip() for t in cvd.split(",") if t.strip()]
    if not all(t.isdigit() for t in tokens):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must contain numeric GPU indices for "
            f"this repo's 2-GPU evolution; got: {cvd!r}"
        )
    if len(tokens) != 2:
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES to list exactly 2 GPUs; "
            f"got {len(tokens)}: {cvd!r}"
        )
    if tokens[0] == tokens[1]:
        raise RuntimeError(
            f"This repo requires 2 distinct GPUs for evolution; "
            f"got CUDA_VISIBLE_DEVICES={cvd!r}"
        )
    return tokens[1]


def get_expected_sglang_uuid() -> Optional[str]:
    """Return the UUID of the GPU that SGLang should be running on (visible GPU #1).

    Primary: NVML device(1).uuid — works for both CVD-set and CVD-unset
    under the repo policy of exactly 2 visible GPUs.
    Fallback (NVML unavailable): nvidia-smi index→uuid map.
    Returns ``None`` if the UUID cannot be determined.
    """
    try:
        from pynvml import (
            nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetUUID, nvmlShutdown,
        )
        nvmlInit()
        try:
            n = nvmlDeviceGetCount()
            if n >= 2:
                handle = nvmlDeviceGetHandleByIndex(1)
                uuid = nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
                return uuid
        finally:
            nvmlShutdown()
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            idx_map: Dict[int, str] = {}
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit():
                    idx_map[int(parts[0])] = parts[1]
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cvd is not None:
                tokens = [t.strip() for t in cvd.split(",") if t.strip()]
                if len(tokens) >= 2:
                    key = int(tokens[1])
                    if key not in idx_map or int(tokens[0]) not in idx_map:
                        LOG.warning(
                            "CVD indices not found in nvidia-smi index list; "
                            "cannot verify GPU pinning in this environment"
                        )
                        return None
                    return idx_map[key]
            elif 1 in idx_map:
                return idx_map[1]
    except Exception:
        pass
    return None


def verify_gpu_pinning(sglang_pid: int) -> None:
    """Verify SGLang PID is on the expected GPU (visible #1) by UUID match.

    Raises RuntimeError on mismatch.  Logs a warning and returns if
    verification is not possible (e.g. NVML unavailable).
    See docs/2XGPU_pod_plan.md §3 / §7.1.
    """
    expected_uuid = get_expected_sglang_uuid()
    if expected_uuid is None:
        LOG.warning("Cannot determine expected SGLang GPU UUID; skipping runtime verification")
        return

    actual_uuid: Optional[str] = None
    try:
        from pynvml import (
            nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetUUID, nvmlDeviceGetComputeRunningProcesses,
            nvmlShutdown,
        )
        nvmlInit()
        try:
            for i in range(nvmlDeviceGetCount()):
                handle = nvmlDeviceGetHandleByIndex(i)
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                if any(p.pid == sglang_pid for p in procs):
                    uuid = nvmlDeviceGetUUID(handle)
                    actual_uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
                    break
        finally:
            nvmlShutdown()
    except Exception:
        pass

    if actual_uuid is None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                for line in out.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == sglang_pid:
                        actual_uuid = parts[1]
                        break
        except Exception:
            pass

    if actual_uuid is None:
        LOG.warning(
            "Could not determine which GPU SGLang (pid %s) is using; "
            "skipping runtime verification", sglang_pid,
        )
        return

    if actual_uuid != expected_uuid:
        raise RuntimeError(
            f"GPU pinning failed: expected embedding on cuda:0 and SGLang "
            f"on cuda:1 (second visible GPU). SGLang pid {sglang_pid} is on "
            f"GPU {actual_uuid}, expected {expected_uuid}."
        )
    LOG.info(
        "Runtime GPU verification passed: SGLang pid %s on GPU %s",
        sglang_pid, actual_uuid,
    )
