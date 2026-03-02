# Plan: Zero-Order Optimization Methods in compressor_2

This document describes the **implementation** of **selected** zero-order optimization methods in the compressor_2 repo, with the ability to **switch between methods via Kconfig**. The objective is **maximized** (fitness); most zero-order libraries **minimize**, so we pass **negative fitness** to them. **Supported methods:** grid_search (range+step discretization), tree-based (Optuna/TPE, SMAC), TR-DFO (BOBYQA/NEWUOA/Py-BOBYQA/PDFO), random_search, gradient-free finite-diff (SPSA, RandomDirection_2pt), population globals (differential_evolution, CMA-ES), and skopt (only when CONFIG_EVAL_DETERMINISTIC=y). Eval-set is **always fixed minibatch** (see §2.2).

**Alignment with current repo:** The plan assumes the **same 2-GPU pod** and evaluation pipeline as the DEAP evolution (see `docs/genetic_evolution_plan.md` and `docs/2XGPU_pod_plan.md`). **Exactly 2 visible GPUs (distinct)** are required; the zero-order path uses the **same 2-GPU enforcement as DEAP** (see §3.4 and §5). **GPU-A** = first visible GPU in the parent's 2-GPU set, **GPU-B** = second. Embedding/generate-processor use **`cuda:0`** (i.e. GPU-A) in processes that see both GPUs. SGLang is **pinned to GPU-B**; because we use a **CVD shim**, the SGLang process sees only that one GPU and therefore uses **`cuda:0`** internally. **SGLang started once and kept running**; no per-eval SGLang restart; processor sent per-request by the runner; **prefetch** of the next point’s processor overlapped with the current eval; **correctness parallelised** with `ThreadPoolExecutor`. Zero-order optimizers **must** use the shared evaluation pipeline (no re-implementation); parallelism is maximized via prefetch and `evaluate_batch` for methods that support it (see §3.2, §4).

**Scope:** The same objective is used everywhere for the supported methods; each is selectable via Kconfig.  
`f = lambda_shortness * shortness_score + lambda_correctness * correctness_ratio` (maximize).

---

## 1. Objective and Evaluation Path

### 1.1 Objective (same as GA)

- **Formula:**  
  `f(x) = lambda_shortness * shortness_score + lambda_correctness * correctness_ratio`
- **shortness_score:** `1 / (1 + mean_tok_len / llm_max_tokens)` (CONFIG_LLM_MAX_TOKENS; over the evaluated set).
- **correctness_ratio:** `num_correct / len(evaluated_set)`.
- **Evaluated set:** a **fixed minibatch** of the pool (CONFIG_EVAL_MINIBATCH_SIZE, CONFIG_EVAL_SEED). Persist evaluation_indices.json and indices_hash—see §2.2.
- **x:** list of `d` floats in **canonical cluster order** (`sorted(cluster_ids, key=int)` from initial deltas). Same list↔dict conversion as GA: `evolution/objective.deltas_list_to_dict` / `deltas_dict_to_list`.

### 1.2 How to evaluate f(x) (shared pipeline only; no re-implementation)

All evaluations go through the **shared** evaluation path. Zero-order is **not allowed** to re-implement these steps; it must call the shared evaluator (see §3.1, §5). The shared path uses `evolution/objective.evaluate_x` (called by **both** GA and zero-order). Do **not** stop or restart SGLang per eval.

1. **Input:** `x` = list of `d` floats in canonical cluster order.
2. **Convert** `x` to deltas dict via `_deltas_list_to_dict(x, cluster_ids)`.
3. **Clip** (if needed): for optimizers without native bounds (e.g. TR-DFO when using a backend that expects box bounds), clip each component of `x` into `[CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH]` inside the objective wrapper **before** step 4; use the clipped list for conversion and evaluation.
4. **Write** deltas to a temp file (e.g. `out_dir / "_eval_deltas.json"` or `_eval_{eval_id}_deltas.json` for traceability).
5. **Generate processor:** run `_generate_processor(cfg, deltas_path, processor_path)` (same subprocess as GA: `compressor_2 generate-processor` on **GPU-A**, with **`env = os.environ`** — no CVD override). In the generate-processor process (which sees both GPUs), we use **`cuda:0`** (the first visible GPU = GPU-A). Use **prefetch** when the next evaluation point is known: wait for in-flight prefetch (if any), **submit prefetch for next x** (if known), then run generate-processor for current x (or use prefetched path). See §1.3.
6. **SGLang:** Do **not** stop or restart SGLang. The server is already running (started once by the driver, same as GA). **Subprocesses inherit the parent's visible set and must not override `CUDA_VISIBLE_DEVICES` — except SGLang.** Only SGLang is allowed to override CVD (CVD shim); all other subprocesses (generate-processor, runner, etc.) must inherit `os.environ` unchanged. In this repo, **SGLang does not support a device flag**, so we pin it via the **CVD shim** in `SGLangServer.start()`. SGLang is **pinned to GPU-B** (the second visible GPU in the parent's 2-GPU set). Because of the shim, the SGLang process sees **only that one GPU** and therefore uses **`cuda:0`** internally. **How the SGLang CVD shim computes its selector:** Let `cvd = os.environ.get("CUDA_VISIBLE_DEVICES")`. If `cvd is None`: SGLang child uses **`CUDA_VISIBLE_DEVICES="1"`** (pin to the second physical GPU on a true 2-GPU host). Else (after validation `len(tokens)==2`): SGLang child uses **`CUDA_VISIBLE_DEVICES=tokens[1]`** (pin to the second token). The selector is a **single numeric index string** and this is the **only** place we set child CVD. See §5.
7. **Runner config:** Build temp YAML via `_build_runner_config(cfg.runner_config_path, evaluation_indices, cfg)` with **evaluation_indices** from EvalContext (fixed minibatch per §2.2).
8. **Run:** `_run_training_set(tmp_runner_cfg, cfg.financebench_jsonl, processor_path)` (financebench_runner subprocess).
9. **Correctness:** `_run_correctness(results, cfg)` — already parallelised with `ThreadPoolExecutor` across examples (no GPU).
10. **Aggregate:** mean_tok_len, num_correct → shortness_score, correctness_ratio → f.
11. **Return:** f (for maximization). When calling a **minimizer**, pass `objective = -f`.

All of this runs **one evaluation at a time** through EvalContext (single SGLang, single runner process). **Parallelism is maximized** when using `evaluate_batch` or `evaluate(x, next_x=...)`: (a) **prefetch** — generate-processor for next x on GPU-A (`cuda:0` in parent) while current x’s training set + correctness run on GPU-B (SGLang server) and CPU; (b) **correctness** — ThreadPoolExecutor over examples; (c) **2-GPU** — no blocking between embedding and SGLang. For **TR-DFO**, when the backend calls the objective internally and does not expose the next x, prefetch is not achievable; see §4.

### 1.3 Prefetch (maximize overlap when next point is known)

- **When next point is known:** Before evaluating `x_i`, wait for the prefetch future (if any). Then **submit** a new prefetch for `x_{i+1}` (deltas + processor) so that generate-processor runs on **GPU-A** (`cuda:0` in the parent) in parallel with the current eval (training set + correctness on GPU-B / CPU). Use non-clashing paths (e.g. `_eval_deltas_prefetch_{c}.json`, `_eval_processor_prefetch_{c}.py` with monotonic counter `c`), same pattern as GA.
- **When next point is not known:** For **TR-DFO**, when the backend calls the objective internally and does **not** expose the next candidate, **prefetch is not achievable**; each eval falls back to **inline** generate-processor (no overlap). Document this explicitly in §4. Methods with batch support (CMA-ES, SPSA, RandomDirection_2pt, grid_search, random_search) use `evaluate_batch` for maximal prefetch. Methods where the library controls the loop (DE, Optuna, SMAC, skopt, TR-DFO) use `ctx.evaluate(x)` without prefetch.
- **Single in-flight prefetch:** One ThreadPoolExecutor (max_workers=1) for prefetch; EvalContext holds one `prefetch_future` at a time.

### 1.4 Temp artifact naming and cleanup

- Temp files: either overwrite `_eval_deltas.json` / `_eval_processor.py` per eval, or include **eval_id** (e.g. `_eval_{eval_id}_deltas.json`) for debugging. **Chosen:** Include eval_id in filenames; **keep only the last N** (e.g. N=5 or configurable) to avoid disk growth; remove older files after each successful eval or when starting a new eval.

### 1.5 Deterministic evaluation (LLM sampling)

- **Recommendation:** **CONFIG_EVAL_DETERMINISTIC=y** is the default recommendation. Fixed minibatch + deterministic decoding (temperature=0, top_p=1, fixed seed) are required for all surrogate-like methods (skopt, Optuna/TPE, SMAC) so that surrogates are not confused by noise.
- **Implementation:** Runner/SGLang decoding settings are reused. **CONFIG_EVAL_DETERMINISTIC** (bool, default y) and **CONFIG_INFERENCE_SEED** (int) are in Kconfig. When CONFIG_EVAL_DETERMINISTIC=y, temperature=0, top_p=1, and fixed seed are forced in the runner config used for evaluation.
- **skopt:** **skopt is allowed only if** CONFIG_EVAL_DETERMINISTIC=y (eval-set is always fixed minibatch). If determinism is off, **disable or forbid** skopt at config load or dispatcher (surrogates get confused by stochastic objectives).
- **grid_search:** **Require** CONFIG_EVAL_DETERMINISTIC=y for grid_search so that points are compared cleanly; if CONFIG_EVAL_DETERMINISTIC is not y, **require or forbid** at config load or dispatcher (recommend require).
- **Generation token budget:** Add **CONFIG_LLM_MAX_TOKENS** (int): maximum number of tokens the LLM may generate per reply to a query. Passed into the runner config (e.g. to the SGLang/generation section) so that all evolution and evaluation runs cap reply length. Required or with a sensible default (e.g. 2048); loader validates value > 0.

### 1.6 Evaluation failure policy

- **Kconfig:** CONFIG_EVAL_TIMEOUT_S (int), CONFIG_EVAL_MAX_RETRIES (int), CONFIG_EVAL_FAILURE_FITNESS (float). On failure: retry up to CONFIG_EVAL_MAX_RETRIES; if all fail, return CONFIG_EVAL_FAILURE_FITNESS (very low value for maximization). **Budget:** Each attempt (including retries and failures) consumes budget (increment n_evals). Best_x / best_f updated **only on successful** evaluations.
- **Logging:** EvalContext logs every attempt in JSONL: `{eval_id, x, status: "ok"|"fail", error?, attempt, indices_hash, timestamp}`; on success include `f`.

---

## 2. Kconfig Additions

### 2.1 New menu: Optimization method

- **CONFIG_OPTIMIZATION_METHOD** (choice):
  - `deap` — current DEAP genetic evolution (default).
  - `grid_search`, `random_search`, `spsa`, `random_direction_2pt`, `differential_evolution`, `cmaes`, `optuna_tpe`, `smac`, `tr_dfo`, `skopt` (skopt allowed only when CONFIG_EVAL_DETERMINISTIC=y; see §2.2, §1.5).
  - Optionally `hybrid` — if kept, defined narrowly as **(global) DE/CMA-ES/Optuna-TPE → (local) TR-DFO** using remaining budget.

When **CONFIG_OPTIMIZATION_METHOD != "deap"**, the loader does **not** require GA-specific options; use `validate_ga=False` and load zero-order options instead.

### 2.2 Eval-set mode (fixed minibatch only)

- **Eval-set is always a fixed minibatch.** Remove `full_pool` and `resampled_minibatch` from Kconfig; the only mode is **fixed_minibatch**.
- **CONFIG_EVAL_MINIBATCH_SIZE** (int): Required. Size of the fixed minibatch.
- **CONFIG_EVAL_SEED** (int): For sampling the minibatch once (and reproducibility). Distinct from CONFIG_OPTIMIZER_SEED.
- **Persist evaluation_indices.json** (indices + seed) and store **indices_hash** for cache keys and resume.
- **skopt:** Allowed **only if** CONFIG_EVAL_DETERMINISTIC=y (fixed minibatch is always true). At config load or dispatcher, if method is skopt and CONFIG_EVAL_DETERMINISTIC is not y, **disable or forbid** skopt (surrogates get confused by stochastic objectives).
- **grid_search:** **Require** CONFIG_EVAL_DETERMINISTIC=y for grid_search (clean comparison of points). At config load or dispatcher, if method is grid_search and CONFIG_EVAL_DETERMINISTIC is not y, **require or forbid** (recommend require).

### 2.3 Options for zero-order methods (all methods except DEAP)

- **CONFIG_ZERO_ORDER_MAX_EVALS** (int): Global hard cap on objective evaluations. EvalContext enforces; when exceeded, raise BudgetExceeded and return best-so-far.
- **CONFIG_EVAL_TIMEOUT_S**, **CONFIG_EVAL_MAX_RETRIES**, **CONFIG_EVAL_FAILURE_FITNESS**: See §1.6.
- **CONFIG_CACHE_ROUND_DECIMALS** or **CONFIG_CACHE_EPS**: Cache key precision; CONFIG_ENABLE_CACHE (bool, optional).
- **CONFIG_EVAL_DETERMINISTIC**, **CONFIG_INFERENCE_SEED** (or CONFIG_SGLANG_SEED): See §1.5.
- **CONFIG_LLM_MAX_TOKENS** (int): Max tokens the LLM may generate per reply; passed to runner config. See §1.5.
- **CONFIG_OPTIMIZER_SEED** (int): For stochastic optimizers (separate from CONFIG_EVAL_SEED).
- **CONFIG_DELTAS_BOUND_LOW**, **CONFIG_DELTAS_BOUND_HIGH** (float): Bounds for clipping and for bounded optimizers.
- **CONFIG_RUN_FINAL_FULL_POOL_EVAL** (bool, default y): After optimizer returns, optionally run one final evaluation of best_x on the full pool (separate from the fixed minibatch used during optimization).
- **Grid search (when CONFIG_OPTIMIZATION_METHOD == "grid_search"):** **CONFIG_GRID_LOW** (float), **CONFIG_GRID_HIGH** (float), **CONFIG_GRID_STEP** (float). Validation: GRID_STEP > 0, GRID_LOW < GRID_HIGH. Compute `n_points = floor((GRID_HIGH - GRID_LOW) / GRID_STEP) + 1` per dimension (float-safe rounding). **CONFIG_GRID_MAX_COMBOS** (int, default e.g. 20000): if `total_combos = n_points ** d` exceeds this limit, **fail fast** with a clear error recommending larger step or smaller range (user can override via Kconfig if intentional). **CONFIG_GRID_ALLOW_TRUNCATION** (bool, default n): if total_combos > CONFIG_ZERO_ORDER_MAX_EVALS, either fail fast (default) or, when y, evaluate in deterministic order and stop at max_evals (return best-so-far). **CONFIG_GRID_BATCH_SIZE** (int, default 64): batch size for streaming grid points to `ctx.evaluate_batch(...)` (performance/memory only).
- Method-specific: CONFIG_TR_DFO_METHOD (e.g. bobyqa, newuoa; PDFO or Py-BOBYQA backend), CONFIG_SKOPT_N_RANDOM_STARTS, CONFIG_OPTUNA_N_TRIALS, CONFIG_ZO_STEP_SIZE, CONFIG_ZO_PERTURB_SCALE, CONFIG_ZO_NUM_DIRECTIONS, CONFIG_ZO_T; if hybrid: CONFIG_HYBRID_GLOBAL_EVALS, CONFIG_HYBRID_LOCAL_EVALS.

Initial point: **CONFIG_INITIAL_DELTAS_PATH** (same as GA). Bounds: [CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH] per dimension.

### 2.4 Loader changes

- In `kconfig_loader.py`: when CONFIG_OPTIMIZATION_METHOD is present and not `"deap"`, set `cfg.optimization_method`, `cfg.zero_order_max_evals`, `cfg.eval_minibatch_size`, `cfg.eval_seed`, `cfg.eval_timeout_s`, `cfg.eval_max_retries`, `cfg.eval_failure_fitness`, `cfg.deltas_bound_low`, `cfg.deltas_bound_high`, `cfg.optimizer_seed`, cache/determinism options, etc. **Eval-set:** Only fixed minibatch; no CONFIG_EVAL_SET_MODE choice (or set it implicitly to fixed_minibatch). Call `_validate_ga_options(cfg)` **only** when `optimization_method == "deap"`. Add `_validate_zero_order_options(cfg)` when method != deap: max_evals >= 1, bounds low < high, minibatch_size set; **if method is skopt and CONFIG_EVAL_DETERMINISTIC is not y, raise or disable skopt**. **If method is grid_search:** validate GRID_LOW < GRID_HIGH, GRID_STEP > 0; compute total_combos = n_points**d; if total_combos > CONFIG_GRID_MAX_COMBOS raise; if total_combos > max_evals and not CONFIG_GRID_ALLOW_TRUNCATION raise; **require CONFIG_EVAL_DETERMINISTIC=y for grid_search**. SPSA/random_direction_2pt: max_evals >= 2. **Environment:** Subprocesses inherit the parent's visible set and must not override `CUDA_VISIBLE_DEVICES` — **except SGLang** (only SGLang may override CVD via the shim in `SGLangServer.start()`); see §5.

---

## 3. Unified Optimizer Interface and Dispatcher

### 3.1 Shared objective evaluator (implemented)

- **Implementation:** `evolution/objective.py` exposes `evaluate_x(x_list, cluster_ids, evaluation_indices, cfg, tokenizer, server_holder, out_dir, sglang_gpu_id, pre_generated_processor_path=None) -> (float, results)`. GA's `ga_driver._evaluate_individual` is a thin wrapper that converts a DEAP individual to a list and calls `evaluate_x`. Zero-order evaluations go through `EvalContext`, which also calls `evaluate_x`. Neither path re-implements generate-processor, SGLang lifecycle, runner config, training set, or correctness.
- **Additional helpers in `evolution/objective.py`:** `generate_processor(cfg, deltas_path, output_path)` (subprocess on `cuda:0`), `build_runner_config(base_config_path, indices, cfg)` (temp YAML), `run_training_set(runner_config_path, financebench_jsonl, processor_path, max_new_tokens=None)` (financebench_runner subprocess), `run_correctness(results, cfg)` (OpenAI correctness via `correctness_openai.evaluate_one`), `deltas_dict_to_list`, `deltas_list_to_dict`, `load_json`, `save_json`.
- **Shared pipeline includes:** generate-processor on the embedding GPU (**GPU-A**; in processes that see both GPUs we use **`cuda:0`**), with **`env = os.environ`** (no CVD override). **Subprocesses inherit the parent's visible set and must not override `CUDA_VISIBLE_DEVICES` — except SGLang.** Only SGLang may override CVD (CVD shim in `SGLangServer.start()`); all other subprocesses (generate-processor, runner, etc.) must inherit `os.environ` unchanged. SGLang server lifecycle (start once, keep running, restart only on crash), correctness ThreadPoolExecutor, prefetch plumbing (`pre_generated_processor_path`). Same non-clashing paths and single in-flight prefetch thread as GA.
- **Wrapper for minimizers:** `make_objective_for_minimizer(...)` returns a callable that clips x into bounds (for methods without native bounds), calls the shared evaluator (via EvalContext), returns **-f**.
- **x0:** Clip into bounds once at initialization for all methods; log clipped x0.

### 3.2 EvalContext and prefetch API (implemented)

- **Module:** `evolution/eval_context.py`.
- **EvalContext** holds: cluster_ids (canonical order), **fixed minibatch** evaluation indices (CONFIG_EVAL_MINIBATCH_SIZE, CONFIG_EVAL_SEED), reference to **server_holder** and SGLang lifecycle (started once by driver), **n_evals** and **max_evals**; **BudgetExceeded** raised when the next eval would exceed max_evals (before starting generate-processor / SGLang for that eval). **best_x**, **best_f**, **best_eval_id** updated only on successful evals.
- **Hard requirement — ordered evaluation with prefetch:** All optimizers must evaluate through EvalContext using one or both of:
  - **`ctx.evaluate(x, next_x=None) -> float`**  
    Wait for prefetch (if any), submit prefetch for `next_x` if provided, then evaluate `x` using `pre_generated_processor_path` from the consumed prefetch. Use when the driver knows the next single point.
  - **`ctx.evaluate_batch(xs: List[x]) -> List[float]`**  
    Runs evaluations **strictly sequentially** (one SGLang, one runner at a time). **Always prefetches** the processor for `xs[i+1]` while evaluating `xs[i]` (training set + correctness). Same non-clashing paths and single in-flight prefetch thread as GA. Use for population/batch methods so they **always** get maximal overlap.
- **Current usage:** CMA-ES (ask/tell), SPSA, RandomDirection_2pt, grid_search, and random_search use `evaluate_batch(xs)` for maximal prefetch. Differential evolution, Optuna/TPE, SMAC, skopt, and TR-DFO use `ctx.evaluate(x)` per point (library controls the loop or next suggestion depends on current result; no prefetch).
- **Failure handling:** Timeout (CONFIG_EVAL_TIMEOUT_S), retries (CONFIG_EVAL_MAX_RETRIES), CONFIG_EVAL_FAILURE_FITNESS on total failure. Log each attempt in JSONL (eval_id, x, status, error, attempt, indices_hash, timestamp).
- **Optional caching:** Cache key (rounded_x_tuple, indices_hash). Enable when CONFIG_EVAL_DETERMINISTIC=y (and CONFIG_ENABLE_CACHE if used). CONFIG_CACHE_ROUND_DECIMALS or CONFIG_CACHE_EPS.
- **History:** JSONL at e.g. `out_dir / "zero_order_history.jsonl"` (eval_id, x, f, indices_hash, timestamp; failures with status "fail").

**Serialization:** Only one evaluation at a time through EvalContext (one SGLang, one runner run). Prefetch runs in a separate thread and does not start a second SGLang or runner.

### 3.3 Optimizer interface

- **Location:** `evolution/optimizers/<method>.py` (one module per method or grouped).
- **Contract:** `run(ctx: EvalContext, x0: List[float], bounds: Tuple[float, float], cfg: EvolutionConfig) -> OptimizerResult`. On early termination (BudgetExceeded or library abort), return `ctx.best_x`, `ctx.best_f` (and history path). All optimizers use **workers=1** / **n_jobs=1** / **parallel=False**; no vectorized objective. Pass library-specific maxiter/maxfun/n_trials where applicable; EvalContext remains the authority for budget.

### 3.4 Dispatcher and 2-GPU enforcement

- **Module:** `evolution/optimize_driver.py`.
- **2-GPU enforcement:** The zero-order path **uses the same 2-GPU enforcement as DEAP**. At entry, call **`validate_2_gpus()`** (from `evolution/gpu_utils.py`, shared by both GA and zero-order). That helper enforces:
  - **If `CUDA_VISIBLE_DEVICES` is unset:** Require `torch.cuda.device_count() == 2`. If `n != 2`, raise `RuntimeError("This repo must run on a 2-GPU pod. Expected exactly 2 visible GPUs; found {n}. If running on a larger host, set CUDA_VISIBLE_DEVICES to exactly two GPUs.")`.
  - **If set:** Parse tokens; require all numeric; require **`len(tokens) == 2`** and **`tokens[0] != tokens[1]`**. If not, raise with the repo’s standard messages (see `docs/2XGPU_pod_plan.md` §2). `validate_2_gpus()` already lives in the shared module `evolution/gpu_utils.py` and is called by both GA and zero-order.
- **Role of optimize_driver:** Load config (validate_ga=False when method != deap), resolve pool and **fixed minibatch** indices, load tokenizer, **start SGLang once** (same as GA, using the same `SGLangServer` and `sglang_gpu_id` from `validate_2_gpus()`), run GPU verification (UUID match for SGLang PID). Build EvalContext. Call the chosen optimizer’s `run(ctx, x0, bounds)`. Persist zero_order_state.json, deltas_best.json, **evaluation_indices.json** (fixed minibatch), final_eval.json when CONFIG_RUN_FINAL_FULL_POOL_EVAL=y; optionally generate processor_best.py **once at end**. Methods that support batch evaluation (CMA-ES, SPSA, RandomDirection_2pt, grid_search, random_search) use `ctx.evaluate_batch(xs)` for prefetch; others use `ctx.evaluate(x)` per point.

---

## 4. Method List and Parallelism Notes

| Method | Eval behavior | Prefetch | Bounded? | Notes |
|--------|----------------|----------|----------|--------|
| grid_search | Exhaustive over discretized box | **Yes** via `ctx.evaluate_batch(...)` | Bounded (range+step) | **Safety:** Explodes combinatorially; use only for tiny d or coarse step. See §4 grid_search note. |
| random_search | Exactly max_evals | **Yes** via `evaluate(x, next_x=...)` or batch | Sample in bounds | Good baseline. |
| spsa | 2 evals/step | **Yes** — **must** use `evaluate_batch([x_plus, x_minus])` each iteration | Clip | Step/perturb Kconfig. |
| random_direction_2pt | 2 evals/step | **Yes** — **must** use `evaluate_batch([x_plus, x_minus])` each iteration | Clip | Finite-diff gradient-free. |
| differential_evolution | pop × iters | **No** — scipy controls the callback loop; `ctx.evaluate(x)` per point | Native | workers=1; updating="immediate". |
| cmaes | pop × iters | **Yes** — ask/tell pattern: `evaluate_batch(candidates)` each generation | Native | workers=1. |
| optuna_tpe | n_trials | **No** — Optuna controls the loop; `ctx.evaluate(x)` per trial | suggest_float | TPE sampler. |
| smac | n_trials | **No** — SMAC controls the loop; `ctx.evaluate(x)` per trial | ConfigurationSpace | SMAC facade.optimize(). |
| tr_dfo | maxfev (BOBYQA/NEWUOA/Py-BOBYQA/PDFO) | **No** — backend calls objective internally; inline generate-processor | Bounds / clip | pybobyqa preferred (NumPy 2 compatible); pdfo fallback. |
| skopt | n_calls | **No** — sequential model-based; next suggestion depends on current result | Native | **Allowed only if** CONFIG_EVAL_DETERMINISTIC=y. ask/tell loop. |
| hybrid (optional) | global then local | Per phase: depends on global method; local TR-DFO no prefetch | — | **(global) DE/CMA-ES/Optuna-TPE → (local) TR-DFO** using remaining budget. |

- **Prefetch alignment:** (1) **CMA-ES:** uses ask/tell — `es.ask()` returns a population, evaluated via `ctx.evaluate_batch(candidates)` each generation, so prefetch runs for every transition. (2) **SPSA / RandomDirection_2pt:** evaluate the ± perturbation pair via **`evaluate_batch([x_plus, x_minus])`** each iteration for maximal prefetch. (3) **grid_search:** enumerate grid in deterministic order; stream points in batches of CONFIG_GRID_BATCH_SIZE and call **`ctx.evaluate_batch(batch_xs)`** for maximal prefetch. (4) **random_search:** pre-generate batches and use `ctx.evaluate_batch(batch)` with batch size 64. (5) **differential_evolution:** scipy controls the callback loop (`_obj(x)` called per point); `ctx.evaluate(x)` without prefetch. (6) **Optuna/SMAC:** library controls the loop; `ctx.evaluate(x)` per trial; next suggestion depends on current result, so no batch/prefetch. (7) **skopt:** ask/tell loop, but next suggestion depends on current result (sequential model-based); `ctx.evaluate(x)` without prefetch. (8) **TR-DFO:** backend calls objective internally; **prefetch is not achievable** — inline generate-processor each eval.
- **Parallelism within one eval:** Prefetch (next processor on **GPU-A**, i.e. `cuda:0` in the parent) + correctness (ThreadPoolExecutor). No parallel **objective** calls (one SGLang, one runner process per eval).
- **skopt:** Allowed **only if** CONFIG_EVAL_DETERMINISTIC=y (and fixed minibatch). Otherwise disable/forbid at config or dispatcher.
- **grid_search — prominent safety note:** The full grid is only feasible for **very small d** or **very coarse step**; otherwise the number of points is exponential (`n_points ** d`). Use grid_search only for small cluster counts or when the search is restricted (coarse step / narrow range). The definition remains "try all points in the discretized matrix"; implement truncation only when CONFIG_GRID_ALLOW_TRUNCATION=y and total_combos > max_evals.

---

## 5. 2-GPU pod and GPU pinning (exact repo policy)

- **Terminology:** **GPU-A** = first visible GPU in the parent's 2-GPU set, **GPU-B** = second. Use **`cuda:0`** / **`cuda:1`** only when referring to a process that **sees both** GPUs (e.g. parent, generate-processor). The **SGLang process** sees **one** GPU (due to the CVD shim) and therefore uses **`cuda:0`** internally.
- **Exactly 2 visible GPUs (distinct):** The repo requires **exactly 2** visible GPUs — not “at least 2”. This is the same for DEAP and zero-order. If `CUDA_VISIBLE_DEVICES` is unset: require `torch.cuda.device_count() == 2`; if not, fail with the repo’s standard message (see `docs/2XGPU_pod_plan.md` §2). If CVD is set: parse tokens; require **`len(tokens) == 2`** and **`tokens[0] != tokens[1]`**; numeric indices only. Both GA and zero-order call `validate_2_gpus()` from `evolution/gpu_utils.py`. The same module also provides `verify_gpu_pinning(sglang_pid)` and `get_expected_sglang_uuid()`.
- **Subprocess CVD rule:** Subprocesses inherit the parent's visible set and must **not** override `CUDA_VISIBLE_DEVICES` — **except SGLang**. Only SGLang is allowed to override CVD (CVD shim); all other subprocesses (generate-processor, runner, etc.) must inherit `os.environ` unchanged.
- **GPU pinning — match repo implementation:**  
  - **Generate-processor / embedding:** Pinned to **GPU-A** (first visible GPU). Subprocess is run with **`env = os.environ`** — **no** CVD override. The generate-processor process sees both GPUs and uses **`cuda:0`** (GPU-A) via device selection (e.g. `--device cuda:0`).
  - **SGLang:** In this repo, **SGLang does not support a device flag**, so we pin it **only** via the **CVD shim** in `SGLangServer.start()` (`evolution/sglang_lifecycle.py`). SGLang is pinned to **GPU-B** (second visible GPU). Because of the shim, the SGLang process sees only that one GPU and uses **`cuda:0`** internally. **CVD shim selector (only place we set child CVD):** Let `cvd = os.environ.get("CUDA_VISIBLE_DEVICES")`. If `cvd is None`: SGLang child uses **`CUDA_VISIBLE_DEVICES="1"`**. Else (after validation `len(tokens)==2`): SGLang child uses **`CUDA_VISIBLE_DEVICES=tokens[1]`**. The selector is a **single numeric index string**. No other subprocess may override CVD.
- **Runtime verification:** In the **parent process** (which has 2 visible GPUs), compute **`expected_uuid = NVML.device(1).uuid`** — this is **GPU-B**. After SGLang starts, verify that SGLang's PID appears on the GPU whose UUID equals **expected_uuid** (via NVML process lists; nvidia-smi fallback). Embedding/generate-processor use GPU-A (`cuda:0` in parent). Same as GA; see `docs/2XGPU_pod_plan.md` §7.1.

---

## 6. Budget Enforcement

- **CONFIG_ZERO_ORDER_MAX_EVALS** is the global hard cap. EvalContext increments n_evals on every objective call and raises **BudgetExceeded** when n_evals == max_evals before starting the next eval. best_x / best_f are always returnable. Each optimizer catches BudgetExceeded and returns ctx.best_x, ctx.best_f.

---

## 7. Bounds and Clipping

- **CONFIG_DELTAS_BOUND_LOW**, **CONFIG_DELTAS_BOUND_HIGH**: Box for all dimensions. For optimizers **without** native bounds (e.g. TR-DFO when using a backend that expects box bounds, or SPSA/random_direction_2pt), clip x in the objective wrapper before evaluation. Canonical order and list↔dict conversion shared (`evolution/objective.deltas_dict_to_list` / `deltas_list_to_dict`).

---

## 8. Dependencies

- **Baseline:** scipy (differential_evolution).
- **TR-DFO:** pdfo (BOBYQA/NEWUOA) or Py-BOBYQA as backend; CONFIG_TR_DFO_METHOD or equivalent to select.
- **Tree-based / surrogate:** optuna (Optuna/TPE), smac (+ ConfigSpace), skopt. On import failure for the selected method, raise a clear error with install hint (e.g. pip install optuna, smac ConfigSpace, scikit-optimize).
- **Population:** cma (CMA-ES). Extras: pip install cma.
- **Determinism and skopt:** skopt allowed only when CONFIG_EVAL_DETERMINISTIC=y; document in installer/README.

---

## 9. Library Usage Details

- **Serial evaluation only:** workers=1, n_jobs=1, parallel=False; no vectorized objective. **grid_search:** Build per-dimension 1D grid `vals = [GRID_LOW + k*GRID_STEP ... <= GRID_HIGH]` (float-safe via `round(..., 10)`); enumerate all points as `itertools.product(vals, repeat=d)` (canonical cluster order); stream in batches of CONFIG_GRID_BATCH_SIZE; call **`ctx.evaluate_batch(batch_xs)`**. Respect max_evals: if total_combos > max_evals and not CONFIG_GRID_ALLOW_TRUNCATION, fail before starting; if truncation allowed, stop at max_evals (BudgetExceeded) and return best-so-far. **random_search:** Pre-generate batches of 64, call `ctx.evaluate_batch(batch)`. **Differential evolution:** `scipy.optimize.differential_evolution` with workers=1, updating="immediate", polish=False; scipy calls `_obj(x)` → `ctx.evaluate(x)` per point (no batch/prefetch). **CMA-ES:** `cma.CMAEvolutionStrategy` ask/tell loop; `es.ask()` → `ctx.evaluate_batch(candidates)` → `es.tell(candidates, [-f for f in fitnesses])`. **Optuna/TPE:** `optuna.create_study(direction="maximize", sampler=TPESampler)` + `study.optimize(_objective, n_trials=...)` where `_objective` calls `ctx.evaluate(x)` per trial. **SMAC:** `HyperparameterOptimizationFacade(scenario, _target)` + `facade.optimize()` where `_target` calls `ctx.evaluate(x)` and returns `-f`. ConfigurationSpace with `Float(f"x{i}", (lo, hi))`. **SPSA / RandomDirection_2pt:** each iteration use **`ctx.evaluate_batch([x_plus, x_minus])`**. **TR-DFO:** tries `pybobyqa.solve(...)` first (NumPy 2 compatible), falls back to `pdfo.pdfo(...)` with `scipy.optimize.Bounds`; no prefetch (inline generate-processor per eval). **skopt:** `skopt.Optimizer` with ask/tell loop; `opt.ask()` → `ctx.evaluate(x)` → `opt.tell([x], [-f])`; only when CONFIG_EVAL_DETERMINISTIC=y.

---

## 10. Output, State, and Resume

- **zero_order_state.json:** method, n_evals_used, max_evals, remaining_evals, best_x, best_f, best_eval_id, optimizer_seed, eval_seed, indices_hash, history_path, final_full_pool_f (when CONFIG_RUN_FINAL_FULL_POOL_EVAL=y). **For grid_search:** also persist grid params for resume/repro: d, GRID_LOW, GRID_HIGH, GRID_STEP, n_points, total_combos, max_evals, truncation (whether CONFIG_GRID_ALLOW_TRUNCATION was used), and log these at start.
- **evaluation_indices.json:** Fixed minibatch (indices + seed); persist and store indices_hash for cache/resume.
- **deltas_best.json:** At end of run (same as GA).
- **processor_best.py:** Generated **once at end** (not per eval).
- **final_eval.json:** When CONFIG_RUN_FINAL_FULL_POOL_EVAL=y (best_x, f_full_pool, shortness_full_pool, correctness_full_pool, indices_hash_full_pool). Do not overwrite best_f with full_pool score.
- **Incremental save and Ctrl+C:** State (zero_order_state.json, deltas_best.json, history, etc.) is **saved during the optimization** (e.g. after each evaluation or at checkpoints) and **on SIGINT (Ctrl+C)**. If the user interrupts with Ctrl+C and runs again, the run **resumes from the same place** (load zero_order_state.json, set x0=best_x, remaining_evals, re-create EvalContext with same indices_hash, continue).

**Resume:** Optuna/SMAC use native persistence; others: load zero_order_state.json, set x0=best_x, remaining_evals=max_evals−n_evals_used, re-create EvalContext with same indices_hash, continue with same method or fallback (e.g. tr_dfo) with remaining budget.

---

## 10.1. Plotting (implemented)

- **Module:** `evolution/graph.py` provides `update_zero_order_fitness_graph(entries, output_dir, method_name)`.
- **Graph:** `zero_order_fitness.png` — scatter of fitness per evaluation index plus a running best-so-far line. For hybrid runs, a vertical dashed line marks the phase-2 (TR-DFO) switch.
- **Callback:** `optimize_driver.py` registers an `on_eval_done(eval_id, f, best_f, n_evals)` callback on EvalContext. The callback accumulates `(eval_id, f, best_f)` entries and calls `update_zero_order_fitness_graph` every 10 evaluations and once at the end (in `finally`).
- **Dependency:** `matplotlib` (Agg backend); if not installed, graph updates are silently skipped.

---

## 10.2. OpenAI API Key Rotation (implemented)

- **Module:** `evolution/openai_key_rotation.py`. Provides `init_keys(filepath, env_var)`, `rotate_to_next_key() -> bool`, `is_openai_api_error(ex) -> bool`, `is_initialized() -> bool`.
- **Initialization:** `evolution/__init__.py` calls `init_keys(cfg.openai_keys_file, cfg.openai_api_key_env)` at the start of `run_evolution()` (before branching to GA or zero-order). The function loads all keys from the file, sets `os.environ[env_var]` to the first key. If the file is missing or empty, raises an error and terminates the run.
- **Rotation:** `correctness_openai.py`, `reflector.py`, and `ga_reflector.py` wrap OpenAI API calls in retry loops. On an OpenAI API error (detected by `is_openai_api_error`), `rotate_to_next_key()` is called. If all keys are exhausted, `RuntimeError("All OpenAI keys exhausted...")` is raised. Non-API errors fall through to existing error handling.
- **Kconfig:** `CONFIG_OPENAI_KEYS_FILE` (string, default `"/workspace/compressor_2/openai_keys.txt"`) — path to a text file with one API key per line.

---

## 11. CLI and single entry point

- **Single entry point invariant:** The only entry for evolution is **`python3 -m compressor_2 evolve --config .config`**. There is no separate zero-order CLI; branching is **inside** the evolve command based on Kconfig.
- **Branch:** If **CONFIG_OPTIMIZATION_METHOD == "deap"**, run **run_ga_evolution(cfg)**. Else run **run_zero_order_evolution(cfg)** (optimize_driver). So `evolve --config` loads config once; then run_evolution(cfg) (or equivalent) branches on cfg.optimization_method.
- **Kconfig validation:** **GA options required only when method == "deap"** (call `_validate_ga_options(cfg)` only then). **Zero-order options required only when method != "deap"** (validate zero-order options and do **not** require GA symbols). Ensures no contradictory validation.
- **Config loading:** When method != deap, load_config(..., validate_ga=False) and run _validate_zero_order_options(cfg) instead.

---

## 12. Python Layout and Shared Code

### 12.1 File layout

- **Shared evaluator (implemented):** `evolution/objective.py` exposes `evaluate_x(...)` which implements the full pipeline: generate-processor (or pre_generated_processor_path), _build_runner_config, _run_training_set, _run_correctness, aggregation. GA's `ga_driver._evaluate_individual` is a thin wrapper calling `evaluate_x`. Zero-order goes through `EvalContext` which also calls `evaluate_x`. Neither path re-implements these steps.
- **evolution/objective.py** — Exposes the shared evaluator and optional clipping wrapper for minimizers. Used by EvalContext and (after refactor) by GA’s _evaluate_individual.
- **evolution/eval_context.py** — EvalContext: cluster_ids, eval-set indices, server_holder ref, n_evals, max_evals, BudgetExceeded, best_x/best_f/best_eval_id, single prefetch future, **evaluate(x, next_x=None)** and **evaluate_batch(xs)**, `on_eval_done` callback. Calls the shared `evaluate_x` only.
- **evolution/optimizers/<method>.py** — Each implements `run(ctx, x0, bounds, cfg) -> OptimizerResult`; returns `ctx.best_x`, `ctx.best_f` on early termination. CMA-ES, SPSA, RandomDirection_2pt, grid_search, and random_search use `ctx.evaluate_batch(xs)` for prefetch. Differential evolution, Optuna/TPE, SMAC, skopt, and TR-DFO use `ctx.evaluate(x)` per point. **evolution/optimizers/grid_search.py** — Build 1D grid per dim (float-safe), Cartesian product via `itertools.product(vals, repeat=d)`, stream batches of CONFIG_GRID_BATCH_SIZE, call `ctx.evaluate_batch(batch_xs)`; respect max_evals and CONFIG_GRID_ALLOW_TRUNCATION.
- **evolution/optimize_driver.py** — Entry for method != deap: calls `validate_2_gpus()` from `gpu_utils.py`, loads config (validate_ga=False), resolves pool/eval indices, tokenizer, starts SGLang once (same `SGLangServer` and pinning as GA), runs `verify_gpu_pinning()`, builds EvalContext with `on_eval_done` plotting callback, dispatches to the chosen optimizer's `run(ctx, x0, bounds, cfg)`, persists state and deltas_best.json, updates `zero_order_fitness.png`, optional full-pool eval and processor_best.py at end. Handles SIGINT for graceful save.
- **evolution/graph.py** — `update_zero_order_fitness_graph(entries, output_dir, method_name)`: fitness scatter + running best-so-far line; optional hybrid phase-switch vertical line.
- **evolution/gpu_utils.py** — `validate_2_gpus()`, `verify_gpu_pinning(sglang_pid)`, `get_expected_sglang_uuid()`. Shared by GA and zero-order.
- **evolution/openai_key_rotation.py** — `init_keys(filepath, env_var)`, `rotate_to_next_key()`, `is_openai_api_error(ex)`, `is_initialized()`. Used by `correctness_openai.py`, `reflector.py`, `ga_reflector.py`.

### 12.2 evolution/__init__.py and entry

- **run_evolution(cfg):** Calls `init_keys(cfg.openai_keys_file, cfg.openai_api_key_env)` first (from `openai_key_rotation`). Then: if `cfg.optimization_method == "deap"`: `run_ga_evolution(cfg)`. Else: `run_zero_order_evolution(cfg)` (lives in optimize_driver.py). **Single entry point** `evolve --config`; branching only inside `run_evolution`.

---

## 13. Implementation Checklist

1. **Kconfig (implemented):** CONFIG_OPTIMIZATION_METHOD (choice: deap, **grid_search**, random_search, spsa, random_direction_2pt, differential_evolution, cmaes, optuna_tpe, smac, tr_dfo, skopt [conditional on CONFIG_EVAL_DETERMINISTIC=y], optionally hybrid), CONFIG_EVAL_MINIBATCH_SIZE, CONFIG_EVAL_SEED, CONFIG_ZERO_ORDER_MAX_EVALS, CONFIG_DELTAS_BOUND_LOW/HIGH, CONFIG_EVAL_TIMEOUT_S, CONFIG_EVAL_MAX_RETRIES, CONFIG_EVAL_FAILURE_FITNESS, CONFIG_OPTIMIZER_SEED, CONFIG_RUN_FINAL_FULL_POOL_EVAL, **CONFIG_LLM_MAX_TOKENS**, CONFIG_EVAL_DETERMINISTIC (default y), cache options, **grid_search when method=grid_search: CONFIG_GRID_LOW, CONFIG_GRID_HIGH, CONFIG_GRID_STEP, CONFIG_GRID_MAX_COMBOS (default 20000), CONFIG_GRID_ALLOW_TRUNCATION (default n), CONFIG_GRID_BATCH_SIZE (default 64)**, method-specific options (CONFIG_TR_DFO_METHOD, etc.). Parser and validation: GA only when method=deap; zero-order when method!=deap; skopt forbidden unless CONFIG_EVAL_DETERMINISTIC=y; **grid_search requires CONFIG_EVAL_DETERMINISTIC=y and grid validation (GRID_STEP>0, GRID_LOW<GRID_HIGH, total_combos ≤ GRID_MAX_COMBOS, and if total_combos>max_evals then GRID_ALLOW_TRUNCATION must be y)**.
2. **EvolutionConfig (implemented):** Fields for optimization_method, zero_order_max_evals, eval_minibatch_size, eval_seed, eval_timeout_s, eval_max_retries, eval_failure_fitness, deltas_bound_low, deltas_bound_high, optimizer_seed, run_final_full_pool_eval, **llm_max_tokens**, cache options, **when method=grid_search: grid_low, grid_high, grid_step, grid_max_combos, grid_allow_truncation, grid_batch_size**, etc. kconfig_loader fills them when method != deap. (Eval-set is always fixed minibatch; no eval_set_mode choice.)
3. **Shared objective (implemented):** `evolution/objective.py` exposes `evaluate_x(x, ...)` with optional pre_generated_processor_path. `make_objective_for_minimizer(...)` clips x and returns -f. Used by EvalContext; GA's `_evaluate_individual` is a thin wrapper.
4. **EvalContext (implemented):** evolution/eval_context.py. Cluster_ids, **fixed minibatch** indices (CONFIG_EVAL_MINIBATCH_SIZE, CONFIG_EVAL_SEED), server_holder, n_evals, max_evals, BudgetExceeded before next eval, best_x/best_f/best_eval_id on success only. Single prefetch future; evaluate(x, next_x=None): wait prefetch, submit prefetch for next_x if provided, call objective with pre_generated_processor_path. Failure handling, logging, optional cache (when CONFIG_EVAL_DETERMINISTIC=y), history JSONL.
5. **Optimizers (implemented):** evolution/optimizers/<method>.py. `run(ctx, x0, bounds, cfg) -> OptimizerResult`; serial evaluation only; on BudgetExceeded/library abort return ctx.best_x, ctx.best_f. Force workers=1, n_jobs=1, etc. Methods with batch support (CMA-ES, SPSA, RandomDirection_2pt, grid_search, random_search) use `ctx.evaluate_batch`; others use `ctx.evaluate(x)` per point.
6. **optimize_driver (implemented):** `validate_2_gpus()` from `gpu_utils.py`, load config (validate_ga=False for method!=deap), **dispatcher:** add grid_search → evolution.optimizers.grid_search.run, resolve pool and **fixed minibatch** indices, tokenizer, start SGLang once, GPU verify, create EvalContext, run optimizer, persist zero_order_state.json (including grid params for grid_search), deltas_best.json, evaluation_indices.json, final_eval.json when applicable, processor_best.py once at end. Prefetch: CMA-ES/SPSA/RandomDirection_2pt/grid_search/random_search use ctx.evaluate_batch(xs); DE/Optuna/SMAC/skopt/TR-DFO use ctx.evaluate(x) per point. Plotting callback (`on_eval_done`) updates zero_order_fitness.png every 10 evals. SIGINT handler for graceful save.
7. **run_evolution (implemented):** In evolution/__init__.py: `init_keys(...)` first, then if `cfg.optimization_method == "deap"`: `run_ga_evolution(cfg)`; else `run_zero_order_evolution(cfg)`.
8. **CLI (no change needed):** `evolve --config` calls run_evolution(cfg).
9. **Dependencies (in requirements.txt):** scipy, matplotlib, optuna, scikit-optimize, smac, ConfigSpace, pdfo, Py-BOBYQA, cma. Clear "install ..." message on import failure for optional deps.
10. **Tests:** Optional unit test for objective with mock; optional integration test with one method and tiny pool.
11. **Plotting (implemented):** `evolution/graph.py` provides `update_zero_order_fitness_graph`. `optimize_driver.py` registers `on_eval_done` callback; updates zero_order_fitness.png every 10 evals and at end.
12. **OpenAI key rotation (implemented):** `evolution/openai_key_rotation.py`. `init_keys` called at start of `run_evolution`. Retry+rotation in `correctness_openai.py`, `reflector.py`, `ga_reflector.py`. CONFIG_OPENAI_KEYS_FILE in Kconfig.

---

## 14. Summary

- **Objective:** Maximize f = lambda_shortness * shortness_score + lambda_correctness * correctness_ratio; same as GA.
- **Methods:** grid_search (range+step discretization; prefetch via evaluate_batch; use only for small d or coarse step), random_search, spsa, random_direction_2pt, differential_evolution, cmaes, optuna_tpe, smac, tr_dfo, skopt (when CONFIG_EVAL_DETERMINISTIC=y), optionally hybrid.
- **2-GPU pod:** **Exactly 2 visible GPUs (distinct)** required. Both GA and zero-order call `validate_2_gpus()` from `evolution/gpu_utils.py`. **GPU-A** = first visible, **GPU-B** = second. Generate-processor: `env = os.environ`, uses **GPU-A** (`cuda:0` in parent). SGLang: **only** SGLang may override CVD — pinned to **GPU-B** via CVD shim in SGLangServer.start() (this repo's SGLang has no device flag); SGLang process sees one GPU and uses `cuda:0` internally. No other subprocess overrides CVD. See §5.
- **Prefetch / shared pipeline:** `evaluate_batch(xs)` always prefetches xs[i+1] while evaluating xs[i]. `evaluate(x, next_x=...)` prefetches when next_x is provided. Methods with prefetch: CMA-ES, SPSA, RandomDirection_2pt, grid_search, random_search (all use `evaluate_batch`). Methods without prefetch: differential_evolution, Optuna/TPE, SMAC, skopt, TR-DFO (library controls the loop or next suggestion depends on current result). GA and zero-order both call `evaluate_x` from `evolution/objective.py`; zero-order does not re-implement it.
- **Unified interface:** EvalContext (evaluate, evaluate_batch), optimizers `run(ctx, x0, bounds, cfg) -> OptimizerResult`. Single entry point: only `evolve --config`; Kconfig validation: GA options only when method=deap, zero-order only otherwise.
- **Eval-set mode:** **Fixed minibatch only** (CONFIG_EVAL_MINIBATCH_SIZE, CONFIG_EVAL_SEED); persist evaluation_indices.json and indices_hash. No full_pool or resampled_minibatch. **skopt** allowed only when CONFIG_EVAL_DETERMINISTIC=y.
- **Budget:** CONFIG_ZERO_ORDER_MAX_EVALS enforced in EvalContext; BudgetExceeded before next eval; best_x/best_f always returnable.
- **Bounds:** CONFIG_DELTAS_BOUND_LOW/HIGH; clip in wrapper for unbounded methods; canonical order and list↔dict shared.
- **Failure:** CONFIG_EVAL_TIMEOUT_S, CONFIG_EVAL_MAX_RETRIES, CONFIG_EVAL_FAILURE_FITNESS; each attempt consumes budget; best updated only on success.
- **Entry:** Single entry `evolve --config` only; if method==deap run GA else zero-order path; save deltas_best.json; processor_best.py once at end.
- **Output:** zero_order_state.json, deltas_best.json, evaluation_indices.json (fixed minibatch), final_eval.json (when CONFIG_RUN_FINAL_FULL_POOL_EVAL=y), zero_order_history.jsonl, zero_order_fitness.png, processor_best.py once at end. **Incremental save and Ctrl+C:** State is saved during the run and on SIGINT so that Ctrl+C then re-run **resumes from the same place** (§10).
- **Plotting:** `evolution/graph.py` updates `zero_order_fitness.png` (fitness scatter + best-so-far line) every 10 evals and at end, via `on_eval_done` callback in `optimize_driver.py`. Hybrid runs show a phase-switch vertical line.
- **OpenAI key rotation:** `evolution/openai_key_rotation.py` loads keys at startup from CONFIG_OPENAI_KEYS_FILE. On API error, rotates to next key; if all exhausted, raises and terminates. Retry logic in `correctness_openai.py`, `reflector.py`, `ga_reflector.py`.
