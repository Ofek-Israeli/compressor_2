# 2×GPU Pod Plan: Embedding on One GPU, LLM on the Other (Mandatory)

**Repo policy: this is not a configurable feature.** Evolution **always** runs on a **2-GPU pod** with **embedding isolated on one GPU** and **SGLang isolated on the other**, with **prefetch always enabled**. No Kconfig options, no single-GPU mode, no CLI device overrides.

---

## 1. Hardware Requirement

- **Mandatory:** **Exactly 2 GPUs** must be visible to the process — either the default host set (exactly 2 GPUs) or the set defined by `CUDA_VISIBLE_DEVICES` (exactly two indices).
- **At startup:** The evolution entrypoint (GA driver init) checks that exactly 2 GPUs are visible; if not, it **raises a hard error** and exits (fail fast). See §2 for the exact error messages. **Operational guidance:** If the host has more than 2 GPUs, users **must** set `CUDA_VISIBLE_DEVICES` to exactly two indices (e.g. `"0,1"` or `"2,3"`) before running evolution or `generate_evolution_processors`.

---

## 2. CUDA_VISIBLE_DEVICES — numeric indices only

`CUDA_VISIBLE_DEVICES` must be a comma-separated list of **integer indices** (e.g. `"0,1"` or `"2,3"`). This repo does **not** support GPU UUID strings here. **Numeric indices only** keeps behavior consistent (e.g. on pods).

The parent must expose **exactly two visible GPUs** — either via unset CVD with exactly 2 GPUs on the host, or via CVD with **exactly 2** numeric tokens and `tokens[0] != tokens[1]`. Within any process (parent or subprocess), **GPU-A is `cuda:0` and GPU-B is `cuda:1`** (logical indices within that process’s visible set). Subprocesses **inherit the same visible set** (do not rewrite `CUDA_VISIBLE_DEVICES` in child env). Each process selects which GPU to use via **framework/device selection** (`cuda:0` vs `cuda:1`), not via rewriting CVD.

- **If parent `CUDA_VISIBLE_DEVICES` is unset** (`cvd is None`):  
  - Require **`torch.cuda.device_count() == 2`** (not ≥2). If `n != 2`, fail fast with:  
    `RuntimeError(f"This repo must run on a 2-GPU pod. Expected exactly 2 visible GPUs; found {n}. If running on a larger host, set CUDA_VISIBLE_DEVICES to exactly two GPUs.")`  
  - Exactly two visible GPUs → embedding uses `cuda:0`, SGLang uses `cuda:1` (device selection only).
- **Else** (CVD set):  
  - Parse `tokens = [t.strip() for t in cvd.split(",") if t.strip()]`.  
  - Validate all tokens numeric; error if any non-numeric.  
  - Require **`len(tokens) == 2`** (not ≥2) **and** `tokens[0] != tokens[1]` (two **distinct** GPUs).  
  - If `len(tokens) != 2`, raise: `RuntimeError(f"Expected CUDA_VISIBLE_DEVICES to list exactly 2 GPUs; got {len(tokens)}: {cvd!r}")`.  
  - If `tokens[0] == tokens[1]`, raise: `RuntimeError(f"This repo requires 2 distinct GPUs for evolution; got CUDA_VISIBLE_DEVICES={cvd!r}")`.  
  - Same as above: embedding uses `cuda:0`, SGLang uses `cuda:1` (device selection; no child CVD override).

**Startup requirement — verified in both cases:**

- **If `CUDA_VISIBLE_DEVICES` is unset:** Compute visible GPU count via `torch.cuda.device_count()` (or NVML). **Fail fast if `n != 2`** with the message above. Do not allow fewer or more than 2.
- **If `CUDA_VISIBLE_DEVICES` is set:** Parse it robustly: split by comma, `strip()` whitespace, drop empty tokens. **Validate that every token is a numeric index** (e.g. `token.isdigit()` or robust int parse). If any token is non-numeric, raise with the numeric-indices message. Require **`len(tokens) == 2` and `tokens[0] != tokens[1]`**. If `len(tokens) != 2`, raise the “Expected … exactly 2 GPUs; got {len(tokens)}” message. If the two tokens are equal, raise the distinct-GPUs message. Treat `CUDA_VISIBLE_DEVICES=""` as **0 visible GPUs**.

**Operational guidance:** If the host has more than 2 GPUs, users **must** set `CUDA_VISIBLE_DEVICES` to exactly two indices (e.g. `"0,1"` or `"2,3"`) before running evolution or `generate_evolution_processors`.

So we never add Kconfig or config fields; we validate the parent’s CVD (and visible count) and pin devices inside each process via **device selection** (`cuda:0` / `cuda:1`), **without** changing `CUDA_VISIBLE_DEVICES` in subprocess env.

---

## 3. Pin Processes via Device Selection (No Child CVD Override)

**Keep the parent’s `CUDA_VISIBLE_DEVICES` unchanged for subprocesses** — pass `os.environ` as-is; do **not** set or override `CUDA_VISIBLE_DEVICES` in the child env. Subprocesses inherit the same two-GPU visible set; pin devices **inside each process** explicitly:

- **Embedding / generate-processor:** Ensure the embedding model (and generate-processor subprocess) uses **`cuda:0`** (first visible GPU in that process). Pass `device="cuda:0"` into the embedder/generate-processor path, or set `torch.device("cuda:0")` there.
- **SGLang:** Launch SGLang with an explicit GPU/device argument so it uses **`cuda:1`** (second visible GPU). Use whatever the repo supports: e.g. `--device cuda:1`, `--gpu-id 1`, or equivalent; if the server does not accept a device flag, use a small shim that sets `CUDA_VISIBLE_DEVICES` once to expose both GPUs and selects `cuda:1` internally. Document the chosen mechanism in code.

- **SGLang** (`evolution/sglang_lifecycle.py` `SGLangServer.start()`): Do **not** override `CUDA_VISIBLE_DEVICES` in Popen env; pass `env = os.environ` (or `{**os.environ}`). Pass device `cuda:1` via SGLang’s supported argument (e.g. `--device cuda:1` / `--gpu-id 1`).
- **generate-processor** (`evolution/ga_driver.py` `_generate_processor(...)`): Run the subprocess with `env = os.environ` (no CVD override). Ensure the subprocess uses `cuda:0` for the embedding model (e.g. pass `device="cuda:0"` or set `torch.device("cuda:0")` in the generate-processor entry path).
- **generate_evolution_processors** (post-evolution): Same — no CVD override; generate-processor uses `cuda:0` inside its process.

No branching on “mode”; we always pin embedding to `cuda:0` and SGLang to `cuda:1` via **device selection**, not env.

**Runtime verification (fail fast):** After starting SGLang and after initializing the embedder / first generate-processor run, **verify** that embedding and SGLang are actually using different devices: embedding must be on `cuda:0`, SGLang’s process must be on GPU-B (the second visible GPU). If embedding is not on `cuda:0` **or** SGLang’s PID is not on the expected GPU for GPU-B, raise `RuntimeError("GPU pinning failed: expected embedding on cuda:0 and SGLang on cuda:1 (second visible GPU).")`.

Use a **single, robust primary method** for `expected_uuid` that does not depend on CVD tokens or `nvidia-smi` index namespaces:

- **Primary (preferred; works for both CVD unset and set):** `expected_uuid` = UUID of **visible GPU #1** in the current process = **`NVML.device(1).uuid`** (i.e. the UUID of NVML device index 1). Under the repo policy of **exactly 2 visible GPUs**, `cuda:1` always corresponds to visible GPU #1, so `expected_uuid` is always NVML device 1’s UUID.
- **Fallback (only when NVML is unavailable):** Use `nvidia-smi`: build index→uuid map via `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`. If **CVD is set**, `expected_uuid = map[int(tokens[1])]` with the existing key-validation (ensure both `tokens[0]` and `tokens[1]` exist as keys; else raise “CVD indices not found in nvidia-smi index list; cannot verify GPU pinning in this environment”). If **CVD is unset**, `expected_uuid = map[1]`.

- **How to implement verification:**  
  - **Embedding:** Confirm the embedding model is created on `torch.device("cuda:0")` (log or return the device at model init). For reporting/debug only, the embedding **physical** GPU is `tokens[0]` when CVD is set (or 0 when unset)—not required for correctness beyond “embedding uses cuda:0”.  
  - **SGLang:** After SGLang starts, query its **PID’s GPU** and verify by **UUID match** to `expected_uuid`:  
    - **NVML:** Get each GPU’s UUID via NVML; find which GPU’s compute-process list contains the SGLang PID; compare that GPU’s UUID to `expected_uuid`.  
    - **nvidia-smi fallback (for PID→UUID when querying SGLang’s GPU):** Map PID → GPU UUID via `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader`; assert the SGLang PID’s UUID equals `expected_uuid` (where `expected_uuid` comes from the primary or fallback above).

---

## 4. Stop/Start Semantics

SGLang is started **once** before the generation loop and **kept running** for the entire evolution run. The logit processor is sent per-request by the runner client (via `custom_logit_processor` in the JSON payload), so the same server instance serves any processor file without restarting.

- **No per-individual restart:** Switching to a new processor does **not** require stopping or restarting SGLang. The runner subprocess loads the processor `.py`, instantiates the class, and sends it in each request payload.
- **Crash recovery:** If the server dies during evolution, `_evaluate_individual` detects it via `is_running()` and auto-restarts before the next training set run.
- SGLang is only stopped during final cleanup (`finally` block) or on crash-recovery restart.

---

## 5. Always-On Prefetch (2-GPU)

Prefetch is **always enabled** (2 GPUs are mandatory). Remove any “two-GPU mode vs single-GPU mode” branching.

- **Design:** Single in-flight prefetch (e.g. a thread or `ThreadPoolExecutor` future) that runs the generate-processor subprocess with **env inherited** (no CVD override); the generate-processor path uses `cuda:0`.
- **Paths must not clash:** Use **non-clashing** paths so that the current eval and the next prefetch never write the same file. Options:
  - Use a **monotonically increasing counter** (e.g. `_eval_processor_prefetch_001.py`, `_eval_deltas_prefetch_001.json`) that increments each time we start a new prefetch; when consuming, use the path that was bound to that prefetch.
  - Or use a **stable ID** for the individual being prefetched (e.g. generation + index or a small hash). Example: `_eval_processor_prefetch_g0_i2.py` and `_eval_deltas_prefetch_g0_i2.json` for generation 0, population index 2.

Choose one scheme and document it in code comments.

- **Wait, submit next, then evaluate (maximum overlap):** At the start of each iteration `i`, wait for the current prefetch future, then **immediately submit** prefetch for individual `i+1` **before** calling `_evaluate_individual(i)`. This way processor generation for `i+1` runs **in parallel** with the current individual's training set + correctness evaluation on the other GPU. If prefetch failed, fall back to inline generate.
- **Cross-generation prefetch:** At the end of each generation, after building the next population, queue a prefetch for `pop[0]` of the next generation. This runs in the background during the persist/logging phase (tree PNG compilation, fitness graph, etc.), so it is ready when the next generation's evaluation loop starts.
- **Offspring[0] prefetch:** After crossover produces the offspring list, immediately queue a prefetch for `offspring[0]` before entering the offspring evaluation loop.
- **Re-eval after mutation:** No prefetch (genome unknown in advance); use `pre_generated_processor_path=None`.

---

## 6. No Optional CLI Device Overrides

Do **not** add `--device` or `--embedding-gpu` to the **evolution** CLI. The split is fixed for this repo’s evolution workflow: embedding on `cuda:0`, SGLang on `cuda:1`, enforced via device selection in code (and SGLang’s supported device flag if any). Embedder and `generate_processor()` use `cuda:0`; SGLang uses `cuda:1` (or equivalent).

---

## 7. Code Changes (By File)

### 7.1 `evolution/ga_driver.py`

- **Helper:** Implement `_validate_2_gpus()` (or keep the name `_get_embedding_and_sglang_gpu_ids()` but change semantics) that **only validates** visibility and distinctness. It does **not** return CVD values for child env. Optionally return a simple struct for use in code, e.g. `(embedding_device="cuda:0", sglang_device="cuda:1")` or `None` (callers use fixed `cuda:0` / `cuda:1`). The helper must:
  - `cvd = os.environ.get("CUDA_VISIBLE_DEVICES")`
  - **If `cvd is None`:**  
    - `n = torch.cuda.device_count()` (or NVML).  
    - Require **`n == 2`** — if `n != 2` raise `RuntimeError(f"This repo must run on a 2-GPU pod. Expected exactly 2 visible GPUs; found {n}. If running on a larger host, set CUDA_VISIBLE_DEVICES to exactly two GPUs.")`.  
    - Return the optional struct if used, or return without value.
  - **Else** (CVD set, including `cvd == ""`):  
    - Parse `tokens = [t.strip() for t in cvd.split(",") if t.strip()]`.  
    - Validate all tokens numeric (e.g. `all(t.isdigit() for t in tokens)`); if any non-numeric, raise `RuntimeError(f"CUDA_VISIBLE_DEVICES must contain numeric GPU indices for this repo's 2-GPU evolution; got: {cvd!r}")`.  
    - Require **`len(tokens) == 2`** (not ≥2) **and** `tokens[0] != tokens[1]`. If `len(tokens) != 2`, raise `RuntimeError(f"Expected CUDA_VISIBLE_DEVICES to list exactly 2 GPUs; got {len(tokens)}: {cvd!r}")`. If `tokens[0] == tokens[1]`, raise `RuntimeError(f"This repo requires 2 distinct GPUs for evolution; got CUDA_VISIBLE_DEVICES={cvd!r}")`.  
    - Return the optional struct if used, or return without value. **Do not return or use CVD strings for child env.**
- **Startup check:** At the start of `run_ga_evolution`, call the helper; it validates (and optionally returns device names). Do **not** pass any CVD override to subprocesses.
- **`_generate_processor`:** Run the subprocess with **`env = os.environ`** (or `{**os.environ}` — no override of `CUDA_VISIBLE_DEVICES`). Ensure the generate-processor entry path uses **`device="cuda:0"`** (or `torch.device("cuda:0")`) for the embedding model.
- **`_evaluate_individual`:**  
  - Add `pre_generated_processor_path: Optional[str] = None`.  
  - **Do not** stop or restart SGLang. The server stays running; the logit processor is sent per-request by the runner client. If the server crashes, auto-restart via `is_running()` check.
  - Generate processor (or use prefetched), run training set, run correctness, compute fitness.
  - Correctness evaluation is parallelised across examples with `ThreadPoolExecutor`.
  - Use non-clashing prefetch paths (counter or stable ID) as above.
- **SGLang startup:** Start SGLang **once** in `run_ga_evolution` before the generation loop. Perform GPU verification (one-time). Keep it running across all evaluations.
- **Prefetch:** Always-on. Spawn prefetch subprocess with **`env = os.environ`** (no CVD override); ensure the prefetch/generate-processor path uses `cuda:0`. Pre-queue prefetch for offspring[0] after crossover; pre-queue prefetch for pop[0] of next gen at end of each generation (overlaps with persist/logging phase).
- **`generate_evolution_processors`:** Call the helper at entry (validates **exactly 2** GPUs; **intentional** — run on same 2-GPU pod, fail fast if not exactly 2). Run `_generate_processor` with **no CVD override**; generate-processor uses `cuda:0` inside.

- **Runtime verification (fail fast):** After SGLang has started and the embedder has been initialized (or after the first generate-processor run), **verify** that embedding is on `cuda:0` and SGLang’s PID is on the **expected GPU for GPU-B** (by **UUID match**). If embedding is not on `cuda:0` **or** SGLang’s PID’s GPU UUID does not match `expected_uuid`, raise `RuntimeError("GPU pinning failed: expected embedding on cuda:0 and SGLang on cuda:1 (second visible GPU).")`.
- **Expected SGLang GPU UUID** — single primary method, nvidia-smi only when NVML unavailable:  
  - **Primary (preferred; works for both CVD unset and set):** `expected_uuid` = **NVML.device(1).uuid** (UUID of visible GPU #1 in the current process). Under exactly 2 visible GPUs, `cuda:1` always corresponds to visible GPU #1.  
  - **Fallback (when NVML is not available):** Build index→uuid map via `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`. If **CVD is set**, `expected_uuid = map[int(tokens[1])]` with key-validation (both `tokens[0]` and `tokens[1]` must exist as keys; else raise “CVD indices not found in nvidia-smi index list; cannot verify GPU pinning in this environment”). If **CVD is unset**, `expected_uuid = map[1]`.
- **How to implement verification:**  
  - **Embedding:** Confirm the embedding model is created on `torch.device("cuda:0")` (already in plan). For debug/reporting only, embedding physical GPU is `tokens[0]` when CVD set (or 0 when unset)—not required for correctness beyond “embedding uses cuda:0”.  
  - **SGLang:** Verify by **UUID match**. **NVML:** Get each GPU’s UUID via NVML; find which GPU’s compute-process list contains the SGLang PID; compare that GPU’s UUID to `expected_uuid`. **nvidia-smi fallback:** PID → GPU UUID (`--query-compute-apps=pid,gpu_uuid`); `expected_uuid` from primary or fallback above; assert SGLang PID’s UUID equals `expected_uuid`.

### 7.2 `evolution/sglang_lifecycle.py`

- **SGLangServer:** Remove `processor_path` from the constructor (unused — the server does not load a processor at startup; processors are sent per-request by the runner client). Pin SGLang to the second GPU via a `CUDA_VISIBLE_DEVICES` shim in `start()`. The server is started once and kept running; it does not need a device argument or processor path.

---

## 8. Main Evolution Plan (§5) Update

In `docs/genetic_evolution_plan.md`, replace **§5 GPU and Parallelism** with:

- **Mandatory exactly 2 GPUs:** Evolution requires **exactly 2** visible GPUs (distinct). At startup, fail fast if not exactly 2. If the host has more than 2 GPUs, set `CUDA_VISIBLE_DEVICES` to exactly two indices before running.
- **Assignment:** Embedding is on GPU-A (`cuda:0`), SGLang is on GPU-B (`cuda:1`). Subprocesses inherit the parent’s visible set (no `CUDA_VISIBLE_DEVICES` override). Pin devices via framework/device selection: generate-processor and embedder use `cuda:0`; SGLang uses `cuda:1` (or equivalent supported flag). Document the SGLang device option in code.
- **Prefetch:** Always on. While evaluating individual i (SGLang + correctness on `cuda:1`), generate-processor for individual i+1 runs on `cuda:0`. Pre-queue prefetch for offspring[0] after crossover and for pop[0] at end of each generation. Use non-clashing paths for prefetched processor/deltas.
- **SGLang lifecycle:** Started once, kept running. No per-individual restart; processors are sent per-request. Auto-restart on crash.
- **Correctness:** Parallelised across examples with `ThreadPoolExecutor`.

---

## 9. Implementation Checklist

1. **ga_driver:** Implement validation helper (e.g. `_validate_2_gpus()` or `_get_embedding_and_sglang_gpu_ids()`): `cvd = os.environ.get("CUDA_VISIBLE_DEVICES")`. **If `cvd is None`:** `n = torch.cuda.device_count()`; require **`n == 2`** (if `n != 2` raise “This repo must run on a 2-GPU pod. Expected exactly 2 visible GPUs; found {n}. If running on a larger host, set CUDA_VISIBLE_DEVICES to exactly two GPUs.”). **Else:** parse `tokens = [t.strip() for t in cvd.split(",") if t.strip()]`; validate all numeric (else raise “must contain numeric GPU indices; got {cvd!r}”); require **`len(tokens) == 2`** (if `len(tokens) != 2` raise “Expected CUDA_VISIBLE_DEVICES to list exactly 2 GPUs; got {len(tokens)}: {cvd!r}”) **and** `tokens[0] != tokens[1]` (else raise “This repo requires 2 distinct GPUs for evolution; got CUDA_VISIBLE_DEVICES={cvd!r}”). **Do not return CVD values for child env**; optionally return `(embedding_device="cuda:0", sglang_device="cuda:1")` for use in code.
2. **ga_driver:** At start of `run_ga_evolution`, call the helper to validate; do **not** override `CUDA_VISIBLE_DEVICES` in subprocess env.
3. **ga_driver:** `_generate_processor`: run subprocess with **`env = os.environ`** (no CVD override). Ensure generate-processor uses **`device="cuda:0"`** (or `torch.device("cuda:0")`) for the embedding model.
4. **ga_driver:** `_evaluate_individual(..., pre_generated_processor_path=...)`: no SGLang stop/start; auto-restart on crash via `is_running()` check; correctness parallelised with `ThreadPoolExecutor`.
5. **ga_driver:** Start SGLang once in `run_ga_evolution` before gen loop; GPU verification once after start.
6. **ga_driver:** Always-on prefetch with non-clashing paths; spawn with **`env = os.environ`**; prefetch/generate-processor uses `cuda:0`. Pre-queue for offspring[0] after crossover; pre-queue for pop[0] at end of gen.
7. **sglang_lifecycle:** Remove `processor_path` from `SGLangServer.__init__`; server started once, kept running. Pin to `cuda:1` via CVD shim in `start()`.
8. **ga_driver:** `generate_evolution_processors`: call the helper at entry (same **exactly 2** GPU requirement; **intentional** — run on same 2-GPU pod, fail fast if not exactly 2); run `_generate_processor` with no CVD override; uses `cuda:0` inside.
9. **Runtime verification:** After SGLang starts (once, before gen loop), verify GPU pinning by UUID match. **Primary:** `expected_uuid` = **NVML.device(1).uuid**. **Fallback:** nvidia-smi index→uuid map. Fail fast on mismatch.
10. **genetic_evolution_plan.md:** Replace §5 with mandatory 2-GPU, SGLang started once, no per-individual restart, correctness parallelised, always-on prefetch with cross-generation and offspring[0] pre-queuing.
11. **No** Kconfig, **no** EvolutionConfig fields, **no** evolution CLI device overrides, **no** single-GPU branch.

---

## 10. Summary

- **Hard requirement:** **Exactly 2** visible GPUs (distinct). Fail fast at startup if not exactly 2: when unset and `n != 2`, use “This repo must run on a 2-GPU pod. Expected exactly 2 visible GPUs; found {n}. If running on a larger host, set CUDA_VISIBLE_DEVICES to exactly two GPUs.”; when set and `len(tokens) != 2`, use “Expected CUDA_VISIBLE_DEVICES to list exactly 2 GPUs; got {len(tokens)}: {cvd!r}”; when the two tokens are equal, use “This repo requires 2 distinct GPUs for evolution; got CUDA_VISIBLE_DEVICES=…”. **Numeric indices only:** if CVD is set and any token is non-integer, fail with “must contain numeric GPU indices; got: …”.
- **Visibility / helper:** `cvd = os.environ.get("CUDA_VISIBLE_DEVICES")`. **Unset** → `n = torch.cuda.device_count()`; require **`n == 2`** (raise with message above if `n != 2`). **Set** → parse, validate numeric, require **`len(tokens) == 2`** (raise “Expected … exactly 2 GPUs; got {len(tokens)}” if not) **and** `tokens[0] != tokens[1]`. Helper **only validates**; it does **not** return CVD values for child env. Optionally return `(embedding_device="cuda:0", sglang_device="cuda:1")`.
- **Operational guidance:** If the host has more than 2 GPUs, users **must** set `CUDA_VISIBLE_DEVICES` to exactly two indices before running evolution or `generate_evolution_processors`.
- **Subprocess env:** **Keep parent’s `CUDA_VISIBLE_DEVICES` unchanged** — pass `os.environ` as-is; do **not** set or override `CUDA_VISIBLE_DEVICES` in child env. Subprocesses inherit the same two-GPU visible set.
- **Device pinning:** Pin **inside each process** via framework/device selection: **embedding / generate-processor** use **`cuda:0`**; **SGLang** use **`cuda:1`** (or equivalent supported flag; document in repo). No CVD rewriting.
- **Runtime verification:** After SGLang starts and embedding initializes (or first generate-processor run), verify embedding is on `cuda:0` and **SGLang’s PID is on the expected GPU for GPU-B by UUID match**. **Primary (preferred; single method for both CVD unset and set):** `expected_uuid` = **NVML.device(1).uuid** — UUID of visible GPU #1 in the current process; under exactly 2 visible GPUs, `cuda:1` always corresponds to visible GPU #1. **Fallback (only when NVML unavailable):** nvidia-smi index→uuid map; if CVD set then `expected_uuid = map[int(tokens[1])]` with validation (both tokens exist as keys); if CVD unset then `expected_uuid = map[1]`. **NVML:** get each GPU’s UUID; find GPU whose compute list contains SGLang PID; compare that GPU’s UUID to `expected_uuid`. **nvidia-smi:** PID→UUID, assert equals `expected_uuid`. If embedding not on `cuda:0` or UUID mismatch, fail fast with `RuntimeError("GPU pinning failed: expected embedding on cuda:0 and SGLang on cuda:1 (second visible GPU).")`. Embedding: confirm model on `torch.device("cuda:0")`; physical GPU for debug/reporting only.
- **generate_evolution_processors:** **Intentional** — part of 2-GPU workflow, run on same 2-GPU pod; uses same helper, fail fast if not **exactly 2** visible GPUs.
- **Prefetch** always on; non-clashing paths; single in-flight prefetch. Pre-queue for offspring[0] after crossover; pre-queue for pop[0] at end of each generation.
- **SGLang lifecycle:** Started once before gen loop, kept running. No per-individual restart; processors sent per-request. Auto-restart on crash.
- **Correctness:** Parallelised across examples with `ThreadPoolExecutor`.
