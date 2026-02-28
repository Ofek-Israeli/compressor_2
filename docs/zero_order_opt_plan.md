# Plan: Zero-Order Optimization Methods in compressor_2

This document is a **comprehensive implementation plan** for adding **all** optimization methods from `docs/zero_order_opt.md` (and the extended list below) to the compressor_2 repo, with the ability to **switch between methods via Kconfig**. The objective is **maximized** (fitness); most zero-order libraries **minimize**, so we pass **negative fitness** to them.

**Scope:** All methods—not only low-\(d\) expensive—are supported; each is selectable via Kconfig. The same objective is used everywhere:  
`f = lambda_shortness * shortness_score + lambda_correctness * correctness_ratio` (maximize).

---

## 1. Objective and Evaluation Path

### 1.1 Objective (same as GA)

- **Formula:**  
  `f(deltas_vector) = lambda_shortness * shortness_score + lambda_correctness * correctness_ratio`
- **shortness_score:** `1 / (1 + mean_tok_len / shortness_scale)` (over the evaluated set).
- **correctness_ratio:** `num_correct / len(evaluated_set)`.
- **Evaluated set:** controlled by **eval-set mode** (Kconfig): full pool, fixed minibatch, or resampled minibatch—see §2.2.

### 1.2 How to evaluate f(x)

1. **Input:** `x` = list of `d` floats in **canonical cluster order** (`sorted(cluster_ids, key=int)` from initial deltas).
2. **Convert** `x` to deltas dict via shared conversion (list ↔ dict using canonical `cluster_ids`).
3. **Clip** (if needed): for optimizers without native bounds (e.g. Nelder–Mead), clip each component of `x` into `[CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH]` inside the objective wrapper before evaluation.
4. **Write** deltas to a temp file (e.g. `out_dir / "_eval_deltas.json"`).
5. **Generate processor:** run `compressor_2 generate-processor` (reuse `_generate_processor` from `ga_driver` or a shared module) to produce `_eval_processor.py`.
6. **SGLang:** stop any running SGLang, start SGLang with the new processor (reuse `SGLangServer` from `sglang_lifecycle`).
7. **Runner config:** build a temp YAML config with `example_indices` = **evaluation-set indices** (from EvalContext: pool or minibatch per §2.2).
8. **Run:** `financebench_runner` with that config and the processor (reuse `_run_training_set`).
9. **Correctness:** run correctness evaluation on all results (reuse `_run_correctness` from `ga_driver`).
10. **Aggregate:** for each result, compute token length; then `mean_tok_len`, `num_correct`, then `shortness_score`, `correctness_ratio`, then `f`.
11. **Return:** `f` (for maximization). When calling a **minimizer**, pass `objective = -f`.

All of this runs **sequentially** per evaluation (one SGLang at a time; one SGLang restart per eval). Reuse existing helpers in `evolution/ga_driver.py` where possible. **Canonical cluster order** and **list↔dict conversion** are shared (e.g. in `evolution/objective.py` or `eval_common.py`).

#### No parallel objective evaluations

- **Constraint:** Evaluation requires generating a processor, restarting a **single** SGLang server, and using the GPU; therefore objective calls must be **strictly serialized**. No hidden parallelism.
- **Plan-level rule:** Even if an optimizer or library supports parallelism, we must force **workers=1** / **n_jobs=1** / **parallel=False** / **vectorized=False**. All evaluations go through EvalContext one at a time.
- Concrete library notes are in §8 (Library Usage Details).

#### Temp artifact naming policy

- Temp files such as `_eval_deltas.json` and `_eval_processor.py` may be either (1) overwritten safely each eval (valid because evaluations are sequential), or (2) include **eval_id** in the filename (e.g. `_eval_{eval_id}_deltas.json`, `_eval_{eval_id}_processor.py`) to improve debugging and postmortems.
- **Chosen approach:** Include **eval_id** in filenames for postmortems; **keep only the last N** such artifacts (e.g. N=5 or configurable) to avoid disk growth. Older files can be removed after each successful eval or when starting a new eval.

#### Deterministic evaluation (LLM sampling)

- **Stochasticity:** Even if **eval_set_mode = fixed_minibatch**, the objective can still be **stochastic** if LLM generation uses nonzero temperature, sampling randomness, or nondeterministic kernels. For optimization and caching to be meaningful, evaluation should be as deterministic as possible.
- **Plan rule:** Prefer **deterministic decoding** (e.g. temperature=0, top_p=1) and a **fixed inference seed** if supported by the runner/SGLang. Reuse existing Kconfig symbols for decoding/sampling if they already exist in the repo; if not, add:
  - **CONFIG_EVAL_DETERMINISTIC** (bool): When true, force deterministic decoding (temperature=0, top_p=1, fixed seed).
  - **CONFIG_INFERENCE_SEED** (int) or **CONFIG_SGLANG_SEED** (int): Fixed seed for inference/LLM generation when deterministic.
  - **CONFIG_TEMPERATURE**, **CONFIG_TOP_P**: Only if not already present elsewhere (for runner/SGLang).
- Interaction with caching is specified in §3.2.

### 1.3 Evaluation failure policy (critical)

When an evaluation **fails** (e.g. SGLang startup failure, runner failure, correctness parsing failure, timeout), the following applies.

- **Kconfig:**
  - **CONFIG_EVAL_TIMEOUT_S** (int): Hard timeout per evaluation (seconds). Abort the eval if exceeded.
  - **CONFIG_EVAL_MAX_RETRIES** (int): How many times to retry the **same** \(x\) after a failure before giving up.
  - **CONFIG_EVAL_FAILURE_FITNESS** (float): Fitness value to return if all retries fail. Use a **very low value** (e.g. \(-10^9\)) since we **maximize**; the optimizer will treat the point as bad and not update best.
- **Budget semantics (chosen policy):** **Each attempt consumes budget**—every call to the objective (including retries and failed attempts) increments the eval counter. This is the **recommended** and **simplest** policy: it avoids infinite loops and keeps budget predictable. (Alternative: only successful evals consume budget would require a separate cap on failed attempts to avoid infinite retry loops; the plan does **not** adopt that.)
- **EvalContext** must **log failures** in the same JSONL (or a dedicated failure log) with fields: `{eval_id, x, status: "ok"|"fail", error, attempt, indices_hash, timestamp}`. On success, log as before with `status: "ok"` and `f`.
- **best_x / best_f / best_eval_id** are updated **only on successful evaluations**. Failed evals (after retries exhausted) do not update best; the optimizer receives CONFIG_EVAL_FAILURE_FITNESS for that \(x\).

---

## 2. Kconfig Additions

### 2.1 New menu: "Optimization method"

- **CONFIG_OPTIMIZATION_METHOD** (choice):
  - `deap` — current DEAP genetic evolution (default for backward compatibility).
  - `grid_search`
  - `random_search`
  - `coordinate_search`
  - `hooke_jeeves` — Hooke–Jeeves / pattern search.
  - `nelder_mead`
  - `powell`
  - `skopt` — GP Bayesian optimization (scikit-optimize).
  - `pdfo` — Trust-region DFO (NEWUOA/BOBYQA).
  - `rbfopt` — RBF surrogate global.
  - `direct` — DIRECT (scipy).
  - `cmaes` — CMA-ES.
  - `differential_evolution`
  - `pso` — Particle swarm.
  - `spsa` — SPSA.
  - `random_direction_1pt` — 1-point random-direction estimator.
  - `random_direction_2pt` — 2-point random-direction estimator.
  - `kiefer_wolfowitz` — Kiefer–Wolfowitz (optional).
  - `cobyla` — COBYLA (if constraints; scipy).
  - `augmented_lagrangian` — Augmented-Lagrangian wrapper (optional, e.g. via NLopt if listed).
  - `optuna` — TPE.
  - `smac` — SMAC3.
  - `hybrid` — global (e.g. DIRECT) then local (pdfo) refinement.

When `CONFIG_OPTIMIZATION_METHOD != "deap"`, the loader does **not** require GA-specific options (population_size, ngen, etc.); use `validate_ga=False` or a separate validation path.

### 2.2 Eval-set mode (determinism vs cost)

- **CONFIG_EVAL_SET_MODE** (choice):
  - `full_pool` — evaluated_set = pool_indices (deterministic, expensive).
  - `fixed_minibatch` — sample once from pool using CONFIG_EVAL_SEED; reuse same indices for all evals (deterministic, cheaper).
  - `resampled_minibatch` — resample minibatch every eval (noisy; document that many optimizers may behave poorly).
- **CONFIG_EVAL_MINIBATCH_SIZE** (int): Required when eval_set_mode is `fixed_minibatch` or `resampled_minibatch`. Size of the minibatch sampled from the pool.
- **CONFIG_EVAL_SEED** (int): Used for fixed_minibatch sampling and for reproducibility of the evaluation set only. (See CONFIG_OPTIMIZER_SEED for optimizer randomness.)
- **Noisy eval-set compatibility:** `resampled_minibatch` is noisy and is **not recommended** for surrogate-based methods (skopt, SMAC, TPE/Optuna, RBFOpt, DIRECT). For those methods, use **fixed_minibatch** or **full_pool**. Plan-level validation: optionally **error or warn** if a surrogate method is selected together with `resampled_minibatch`.
- **Persist evaluation-set indices for reproducibility:**
  - **fixed_minibatch:** Write **evaluation_indices.json** under `out_dir` containing the sampled indices and the seed used. Compute and store its SHA/hash as **indices_hash** (for cache keys and history).
  - **full_pool:** Persist the **resolved pool indices** (or at least the hash + source config, e.g. CONFIG_FINANCEBENCH_POOL_INDICES) so runs are reproducible.
  - **resampled_minibatch:** Log the **actual indices per eval** in history (or at least **indices_hash** for that eval) so results are traceable.

- **Optional multi-fidelity evaluation (default off):** When **CONFIG_USE_MULTI_FIDELITY** (bool, default n) is enabled, search uses a **cheap** objective and confirmation uses an **expensive** objective. All multi-fidelity options are ignored when CONFIG_USE_MULTI_FIDELITY=n; the plan remains valid without them.
  - **CONFIG_SEARCH_EVAL_SET_MODE** = fixed_minibatch | full_pool (default **fixed_minibatch**): evaluation set for the **search** (cheap) objective used by the optimizer on every call.
  - **CONFIG_SEARCH_MINIBATCH_SIZE** (int): Required when CONFIG_SEARCH_EVAL_SET_MODE uses a minibatch.
  - **CONFIG_CONFIRM_EVAL_SET_MODE** = full_pool | fixed_minibatch (default **full_pool**): evaluation set for the **confirmation** (expensive) objective, run only occasionally.
  - **CONFIG_CONFIRM_MINIBATCH_SIZE** (int): Required when CONFIG_CONFIRM_EVAL_SET_MODE uses a minibatch.
  - **CONFIG_CONFIRM_EVERY_K_EVALS** (int, default e.g. 10): run a confirmation evaluation every K successful search evals (or every K candidate improvements; define precisely in EvalContext).
  - **CONFIG_CONFIRM_IMPROVEMENT_EPS** (float, default 0.0): only run confirmation when the search-score improves over current best by at least eps.
  - **Seeds for confirmation:** Confirmation uses the **same** eval determinism settings (temperature, inference seed) as search; it has its **own indices hash** (from the confirm eval set), persisted for reproducibility.

### 2.3 Options used by zero-order methods (all methods except DEAP)

- **CONFIG_ZERO_ORDER_MAX_EVALS** (int): **Global hard cap** on objective evaluations. Default 100. EvalContext enforces this; when exceeded it raises BudgetExceeded and the driver returns best-so-far.
- **CONFIG_EVAL_TIMEOUT_S** (int): Hard timeout per evaluation (seconds). Default e.g. 600.
- **CONFIG_EVAL_MAX_RETRIES** (int): Number of retries for the same \(x\) after a failure. Default e.g. 2.
- **CONFIG_EVAL_FAILURE_FITNESS** (float): Fitness to use if all retries fail (we maximize; recommend e.g. \(-10^9\)).
- **CONFIG_CACHE_ROUND_DECIMALS** (int) or **CONFIG_CACHE_EPS** (float): Precision for cache key and duplicate detection. See §3.2.
- **CONFIG_ENABLE_CACHE** (bool, optional): Explicitly enable caching when evaluation is stochastic; when unset, cache is gated by determinism (see §3.2).
- **Evaluation determinism (decoding):** Reuse existing runner/SGLang decoding symbols if present; otherwise add **CONFIG_EVAL_DETERMINISTIC** (bool), **CONFIG_INFERENCE_SEED** or **CONFIG_SGLANG_SEED** (int), and optionally **CONFIG_TEMPERATURE**, **CONFIG_TOP_P**—see §1.2 “Deterministic evaluation (LLM sampling)”.
- **CONFIG_OPTIMIZER_SEED** (int): Used by **all stochastic optimizers** (random search, differential evolution, CMA-ES, PSO, SPSA perturbations, Optuna, SMAC, etc.) for reproducibility. Kept **separate** from CONFIG_EVAL_SEED (which is only for fixed_minibatch sampling).
- **CONFIG_DELTAS_BOUND_LOW** (float): Lower bound for each delta (same for all dimensions). Default -2.0.
- **CONFIG_DELTAS_BOUND_HIGH** (float): Upper bound for each delta. Default 2.0.
- **Bounds and clipping:** For optimizers **without** native bounds (e.g. Nelder–Mead, Powell), the objective wrapper **clips** proposed `x` into `[CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH]` before evaluation. Canonical cluster order and list↔dict conversion are shared.

Method-specific (examples):
- **CONFIG_PDFO_METHOD** (string): When method=pdfo, `"newuoa"` or `"bobyqa"`. Default `bobyqa` for bound constraints. Use `options={"maxfev": CONFIG_ZERO_ORDER_MAX_EVALS}`.
- **CONFIG_SKOPT_N_RANDOM_STARTS** (int): Random starts for skopt. Default 5.
- **CONFIG_OPTUNA_N_TRIALS** (int): Trials for optuna (or use MAX_EVALS).
- **CONFIG_HYBRID_GLOBAL_EVALS** (int): Evaluations for global phase when method=hybrid. Default 30.
- **CONFIG_HYBRID_LOCAL_EVALS** (int): Max evaluations for local refinement (pdfo). Default 100.
- **CONFIG_DIRECT_LEN_TOL** / **CONFIG_DIRECT_VOL_TOL** (optional): Tune DIRECT behavior; primary budget knob is `maxfun=CONFIG_ZERO_ORDER_MAX_EVALS` via scipy.optimize.direct.
- **Gradient-estimate / ascent methods (SPSA, random-direction, Kiefer–Wolfowitz):**
  - **CONFIG_ZO_STEP_SIZE** (float) or **CONFIG_SPSA_A** / **CONFIG_SPSA_ALPHA** for step/learning rate (e.g. \(\eta_t\) in \(x_{t+1} = \text{clip}(x_t + \eta_t \hat{g}_t)\)).
  - **CONFIG_ZO_PERTURB_SCALE** (float): perturbation scale \(\sigma\) or \(c\) for finite-difference / SPSA.
  - **CONFIG_ZO_NUM_DIRECTIONS** (int): number of random directions per step (for random-direction estimators).
  - **CONFIG_ZO_T** (int) or derive iterations from max evals: e.g. SPSA uses 2 evals/iter → \(T = \lfloor \text{max\_evals}/2 \rfloor\).
- **Final full-pool evaluation:** **CONFIG_RUN_FINAL_FULL_POOL_EVAL** (bool, default y): After the optimizer returns, run one final evaluation of best_x on **full_pool** unless eval_set_mode was already full_pool. (Optional) **CONFIG_FINAL_EVAL_SET_MODE** = full_pool | fixed_minibatch for flexibility later.
- **Multi-fidelity (optional):** When CONFIG_USE_MULTI_FIDELITY=y, also parse CONFIG_SEARCH_EVAL_SET_MODE, CONFIG_SEARCH_MINIBATCH_SIZE, CONFIG_CONFIRM_EVAL_SET_MODE, CONFIG_CONFIRM_MINIBATCH_SIZE, CONFIG_CONFIRM_EVERY_K_EVALS, CONFIG_CONFIRM_IMPROVEMENT_EPS (see §2.2).

Initial point for all methods: from **CONFIG_INITIAL_DELTAS_PATH** (same as GA). Bounds: [CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH] per dimension.

### 2.4 Loader changes and method-specific validation

- In `kconfig_loader.py`, add parsing for the new symbols. When `CONFIG_OPTIMIZATION_METHOD` is present and not `"deap"`, set `cfg.optimization_method`, `cfg.zero_order_max_evals`, `cfg.optimizer_seed`, `cfg.deltas_bound_low`, `cfg.deltas_bound_high`, `cfg.eval_set_mode`, `cfg.eval_minibatch_size` (when applicable), `cfg.eval_seed`, `cfg.eval_timeout_s`, `cfg.eval_max_retries`, `cfg.eval_failure_fitness`, `cfg.cache_round_decimals` or `cfg.cache_eps`, `cfg.enable_cache`, `cfg.eval_deterministic`, `cfg.inference_seed` or `cfg.sglang_seed`, `cfg.run_final_full_pool_eval`, `cfg.final_eval_set_mode` (optional), etc. Call `_validate_ga_options(cfg)` only when `optimization_method == "deap"`.
- **Method-specific validation rules (loader-level):**
  - **SPSA / 2-point / Kiefer–Wolfowitz:** Require **CONFIG_ZERO_ORDER_MAX_EVALS >= 2** and ensure derived **T >= 1** (e.g. for SPSA \(T = \lfloor \text{max\_evals}/2 \rfloor\)).
  - **CONFIG_ZO_NUM_DIRECTIONS >= 1** when the method uses it (random-direction estimators).
  - **CONFIG_CACHE_ROUND_DECIMALS:** if used, ensure within a sane range (e.g. **0–12**); if **CONFIG_CACHE_EPS** is used, ensure **EPS > 0**.
  - **CONFIG_EVAL_SET_MODE = fixed_minibatch** (or resampled_minibatch): require **CONFIG_EVAL_MINIBATCH_SIZE <= len(pool_indices)** (validate after pool is resolved).

---

## 3. Unified Optimizer Interface and Dispatcher

### 3.1 Shared objective evaluator

- **Module:** `evolution/objective.py`.
- **Role:** Shared evaluator returning **fitness to maximize**.
- **Function:**  
  `evaluate_objective(cfg, cluster_ids, evaluation_indices, x: List[float], *, tokenizer, server_holder, out_dir: Path) -> float`
  - `x`: list of length `len(cluster_ids)` in **canonical order**.
  - Returns fitness: `lambda_shortness * shortness_score + lambda_correctness * correctness_ratio`.
- **Objective callable contract:** The objective wrapper (used by EvalContext and passed to minimizers) must **accept** `x` as **list or numpy array**; **convert to list of floats**; **clip** into bounds before evaluation. All callers see the same contract.
- **Wrapper for minimizers:**  
  `make_objective_for_minimizer(...)` returns a callable that (optionally clips `x` into bounds for methods without native bounds,) calls `evaluate_objective` and returns **-f**.
- **x0 handling:** **Clip x0 into bounds once at initialization** for all methods (before passing to any optimizer). **Log the clipped x0** (e.g. in zero_order_state.json or startup log) so runs are reproducible and out-of-bounds initial points are visible.

Implementation: convert `x` to deltas dict via shared canonical list↔dict; write deltas; call shared helpers (generate processor, build runner config for `evaluation_indices`, run training set, run correctness, aggregate). Reuse from `ga_driver` where possible. **Canonical cluster order** is defined once (e.g. `sorted(cluster_ids, key=int)`) and used everywhere.

### 3.2 EvalContext

- **Module:** `evolution/eval_context.py` (or similar).
- **EvalContext** owns:
  - **cluster_ids** in canonical order.
  - **Evaluation-set indices:** pool or minibatch (depending on CONFIG_EVAL_SET_MODE); for fixed_minibatch, sample once with CONFIG_EVAL_SEED; for resampled_minibatch, resample each time (noisy).
  - **SGLang lifecycle handle** (or reference to server_holder / start/stop).
  - **Eval counter + hard budget enforcement:** increment `n_evals` on each objective call. **Budget semantics (precise):** If `n_evals == max_evals`, the **next** attempted evaluation triggers **BudgetExceeded before** starting SGLang / generating a processor—so no partial eval counts toward the cap without a returnable best. Do **not** rely on library-specific maxiter/maxfun/n_trials as the sole cap—still pass them where applicable, but EvalContext is the authority.
  - **Best solution (guaranteed returnable):** EvalContext must maintain **best_x**, **best_f**, and **best_eval_id** updated **only on successful evaluations** (after computing \(f\)). Failed evals (and CONFIG_EVAL_FAILURE_FITNESS) do not update best. Thus on any early termination (BudgetExceeded or library abort), the driver can always return a valid best.
  - **Failure logging:** Log every attempt in JSONL with fields: `{eval_id, x, status: "ok"|"fail", error, attempt, indices_hash, timestamp}`. On success include `f`; on failure include `error` and `attempt` (retry index).
  - **Optional caching:** Cache key is **(rounded_x_tuple, indices_hash)** where **indices_hash** identifies the exact evaluation indices used. **rounded_x_tuple** is defined precisely: round each coordinate of `x` to **N decimal places** (from CONFIG_CACHE_ROUND_DECIMALS) or using CONFIG_CACHE_EPS for tolerance; this tuple is used for **caching** and for **duplicate detection** in some methods. **Cache precision:** CONFIG_CACHE_ROUND_DECIMALS (int) or CONFIG_CACHE_EPS (float) makes cache precision configurable. **Determinism and caching:** Caching is **only enabled** when evaluation is **deterministic** (CONFIG_EVAL_DETERMINISTIC or equivalent, e.g. temperature=0 and fixed inference seed) **and** eval_set_mode is **not** resampled_minibatch. If evaluation is stochastic (e.g. temperature > 0 or nondeterministic kernels), **disable caching by default** unless **CONFIG_ENABLE_CACHE** is set. **Rule for resampled_minibatch:** **caching is disabled by default** in that mode unless the cache is keyed by per-eval indices hash (so each resampled minibatch is keyed separately); document this explicitly.
  - **History logging (JSONL):** path e.g. `out_dir / "zero_order_history.jsonl"`; each line: `{eval_id, x, f, set_mode, indices_hash, timestamp}` for successes; failures use the same stream or a dedicated one with `status: "fail"` and the fields above. **Add to each line (or a header):** decoding settings and **inference seed** (or a short hash of them) so runs are auditable. When **multi-fidelity** is enabled, add **fidelity: "search"|"confirm"** and keep **separate indices_hash** per fidelity (search_indices_hash, confirm_indices_hash) so each line is unambiguous.

- **Multi-fidelity semantics (when CONFIG_USE_MULTI_FIDELITY=y):**
  - **Two evaluators:** EvalContext exposes **evaluate_search(x)** — used by the optimizer on **every** call (cheap, CONFIG_SEARCH_EVAL_SET_MODE / CONFIG_SEARCH_MINIBATCH_SIZE) — and **evaluate_confirm(x)** — used **only occasionally** (expensive, CONFIG_CONFIRM_EVAL_SET_MODE / CONFIG_CONFIRM_MINIBATCH_SIZE).
  - **Budget policy:** Both search and confirmation evaluations consume the **same global budget** (CONFIG_ZERO_ORDER_MAX_EVALS). Optionally a second cap (e.g. max_confirm_evals) can be added later; the plan does not require it.
  - **Best tracking (important):** Track **best_search_x** / **best_search_f** (best so far on the search objective) and **best_confirm_x** / **best_confirm_f** (the “true best” on the confirm objective). **Official best_x / best_f** are updated **only when a confirmation succeeds** (so the reported best is the confirm-validated point). When multi-fidelity is **disabled**, the single evaluator remains the source of truth and best_x/best_f behave as today.
  - Optimizers call the **search** objective only; EvalContext may trigger confirm evaluations internally according to CONFIG_CONFIRM_EVERY_K_EVALS and CONFIG_CONFIRM_IMPROVEMENT_EPS.

All objective calls go through EvalContext so that budget and history are centralized. **Serialization:** EvalContext ensures exactly one evaluation at a time; no parallel or vectorized objective calls (see §1.2 “No parallel objective evaluations”).

### 3.3 Optimizer interface

- **Location:** `evolution/optimizers/<method>.py` (one module per method, or a few grouped).
- **Contract:** Each optimizer implements a function that conforms to:
  - **Input:** `run(ctx: EvalContext, x0: List[float], bounds: Tuple[float, float]) -> OptimizerResult`
  - **Output:** `OptimizerResult(best_x, best_f, history_path, [method_specific_state_path])`
- **Early termination:** On **any** early termination (BudgetExceeded or library abort), each driver must return **ctx.best_x** and **ctx.best_f** (so the returnable best is always the one tracked by EvalContext). Library-specific limits (maxiter, maxfun, n_trials) are still passed to the library but are not the sole enforcement mechanism.

### 3.4 Dispatcher

- **Module:** `evolution/optimize_driver.py`.
- **Role:** Dispatch by `cfg.optimization_method`; load EvalContext; call the chosen optimizer’s `run(ctx, x0, bounds)`; persist result (deltas_best.json, zero_order_state.json, optional processor_best.py once at end).

---

## 4. Method List: All Methods from zero_order_opt.md and Extended Set

For each method: expected eval-count behavior and whether it is realistic given **one SGLang restart per eval**.

| Method | Driver module | Expected eval behavior | Realistic (1 SGLang/eval)? | Bounded? | Noisy sensitivity | Kconfig / notes |
|--------|----------------|-------------------------|-----------------------------|----------|--------------------|------------------|
| **Grid search** | `grid_search.py` | \(O(M^d)\); only for very small \(d\) and small M | Only tiny grids | Native (discrete grid in bounds) | N/A | CONFIG_GRID_POINTS_PER_DIM or similar; use sparingly. |
| **Random search** | `random_search.py` | Exactly CONFIG_ZERO_ORDER_MAX_EVALS | Yes | Sample in bounds | Moderate | Simple; good baseline. |
| **Coordinate search** | `coordinate_search.py` | \(O(d \cdot \text{iters})\) per cycle | Yes if iters capped | Native (line search in bounds) | Low | Cycle over dimensions; can cap line-search evals. |
| **Hooke–Jeeves / pattern search** | `hooke_jeeves.py` | Pattern + exploratory moves; evals scale with d and steps | Yes with budget | Clip in wrapper or native if impl has bounds | Low | scipy or custom; clip if no bounds. |
| **Nelder–Mead** | `nelder_mead.py` | Simplex; often 2–3×d to hundreds | Yes | **No native bounds** → clip in wrapper | Sensitive to noise | scipy.optimize.minimize(..., options={"maxfev": ...}); clip in wrapper. |
| **Powell** | `powell.py` | Conjugate directions; similar scale to NM | Yes | **No native bounds** → clip in wrapper | Sensitive | scipy.optimize.minimize(..., options={"maxfev": ...}); clip in wrapper. |
| **Bayesian (skopt)** | `skopt.py` | n_calls = budget; n_random_starts + model-based | Yes | Native (dimensions) | GP handles moderate noise | CONFIG_SKOPT_N_RANDOM_STARTS. **Surrogate:** not recommended with resampled_minibatch. |
| **Trust-region DFO (pdfo)** | `pdfo.py` | maxfev; BOBYQA for bounds | Yes | **BOBYQA** for bounds; use `method="bobyqa"`, `options={"maxfev": ...}` | Prefers deterministic | CONFIG_PDFO_METHOD; bobyqa appropriate for bound constraints. |
| **RBFOpt** | `rbfopt.py` | max_evaluations | Yes | Native (var_lower/var_upper) | Moderate | RbfoptSettings(max_evaluations=...). **Surrogate:** not recommended with resampled_minibatch. |
| **DIRECT** | `direct.py` | **maxfun** is primary budget knob; optionally len_tol/vol_tol | Yes | Native (Bounds) | Deterministic | scipy.optimize.direct(func, bounds, maxfun=CONFIG_ZERO_ORDER_MAX_EVALS). **Surrogate:** not recommended with resampled_minibatch. |
| **CMA-ES** | `cmaes.py` | Population × generations; control by budget | Yes with population size and maxiter | Clip or use bounds in cma if supported | Can handle some noise | cma package; pass maxfun/budget; clip if needed. |
| **Differential evolution** | `differential_evolution.py` | popsize × maxiter; scipy | Yes | Native (bounds) | Robust to noise | Set maxiter/popsize so popsize×maxiter fits budget; EvalContext enforces true cap. |
| **PSO** | `pso.py` | n_particles × iters | Yes | Clip or bounds in pyswarms | Moderate | pyswarms; optional. |
| **SPSA** | `spsa.py` | 2 evals per iteration (gradient approx) | Yes | Clip in wrapper | Designed for noisy | Iterative ascent: \(x_{t+1} = \text{clip}(x_t + \eta_t \hat{g}_t)\). CONFIG_ZO_STEP_SIZE / CONFIG_SPSA_A, CONFIG_SPSA_ALPHA; CONFIG_ZO_PERTURB_SCALE; \(T = \lfloor \text{max\_evals}/2 \rfloor\). Rademacher ±1 perturbations. |
| **Random-direction 1-pt** | `random_direction_1pt.py` | 1 eval per step | Yes | Clip in wrapper | Noisy | Same ascent rule; 1-pt gradient estimator. CONFIG_ZO_NUM_DIRECTIONS, CONFIG_ZO_PERTURB_SCALE, CONFIG_ZO_STEP_SIZE, CONFIG_ZO_T or from max_evals. |
| **Random-direction 2-pt** | `random_direction_2pt.py` | 2 evals per step | Yes | Clip in wrapper | Noisy | \(\hat{g} = (f(x+\sigma u) - f(x-\sigma u))/(2\sigma) \cdot u\); then \(x_{t+1} = \text{clip}(x_t + \eta_t \hat{g})\). CONFIG_ZO_NUM_DIRECTIONS, CONFIG_ZO_PERTURB_SCALE, CONFIG_ZO_STEP_SIZE. |
| **Kiefer–Wolfowitz** | `kiefer_wolfowitz.py` | 2 evals per step (2-sided) | Yes | Clip | Noisy | Same as 2-pt estimator; optional step schedule. CONFIG_ZO_PERTURB_SCALE, CONFIG_ZO_STEP_SIZE, CONFIG_ZO_T. |
| **COBYLA** | `cobyla.py` | Scipy; constraint-based | Yes if constraints used | **Preferred:** box bounds as inequality constraints (see §6) | Sensitive | Box-bounded only unless CONFIG_USE_CONSTRAINTS=y; if no extra constraints, warn and run with bounds-as-ineq. |
| **Augmented-Lagrangian** | `augmented_lagrangian.py` | NLopt or scipy; for constraints | Yes | Via constraints | Optional / future; box-bounded only unless CONFIG_USE_CONSTRAINTS=y. |
| **Optuna (TPE)** | `optuna.py` | n_trials | Yes | suggest_float in bounds | Handles noise | CONFIG_OPTUNA_N_TRIALS. **Surrogate (TPE):** not recommended with resampled_minibatch. |
| **SMAC** | `smac.py` | n_trials in Scenario | Yes | ConfigurationSpace Float(..., bounds) | Handles noise | **ConfigurationSpace(seed=...)** with **Float(...)**; incumbent via **smac.validate(incumbent)**. **Surrogate:** not recommended with resampled_minibatch. |
| **Hybrid (global → local)** | `hybrid.py` | CONFIG_HYBRID_GLOBAL_EVALS + CONFIG_HYBRID_LOCAL_EVALS | Yes | Same as DIRECT + pdfo | Depends on stages | Run global (e.g. DIRECT) then pdfo from best_x. |

### 4.1 Gradient-estimate methods (SPSA, random-direction, Kiefer–Wolfowitz)

These methods implement **iterative ascent**: \(x_{t+1} = \text{clip}(x_t + \eta_t \hat{g}_t)\) where \(\eta_t\) is the step size and \(\hat{g}_t\) is a gradient estimate.

- **2-point (random-direction):** \(\hat{g} = \frac{f(x+\sigma u) - f(x-\sigma u)}{2\sigma} \cdot u\), where \(u\) is a random direction (e.g. unit or Gaussian). Uses 2 evals per step.
- **SPSA:** Same idea with **Rademacher** ±1 perturbations (each component of the perturbation is ±1); 2 evals per iteration. Step size and perturbation scale typically decay with \(t\) (e.g. CONFIG_SPSA_A, CONFIG_SPSA_ALPHA).
- **1-point:** One eval per step (e.g. forward or backward difference); noisier but half the evals per step.
- **Kiefer–Wolfowitz:** Typically 2-sided (2 evals/step) with a prescribed step schedule; same 2-pt formula.

Required Kconfig (see §2.3): CONFIG_ZO_STEP_SIZE (or CONFIG_SPSA_A / CONFIG_SPSA_ALPHA), CONFIG_ZO_PERTURB_SCALE (\(\sigma\)/c), CONFIG_ZO_NUM_DIRECTIONS (for random-direction), CONFIG_ZO_T or derive from max_evals (e.g. SPSA: \(T = \lfloor \text{max\_evals}/2 \rfloor\)).

---

## 5. Budget Enforcement (Centralized)

- **CONFIG_ZERO_ORDER_MAX_EVALS** is the **global hard cap**.
- **EvalContext** increments `n_evals` on every objective evaluation and **raises BudgetExceeded** when the next eval would exceed the cap (i.e. before starting SGLang/generate-processor for that eval). EvalContext maintains **best_x**, **best_f**, **best_eval_id** on every successful eval so a returnable best is always available.
- Each optimizer **driver** catches BudgetExceeded (and any library abort) and **returns ctx.best_x / ctx.best_f** (and history path).
- Library-specific parameters (maxiter, maxfun, n_trials) are still **passed** where applicable, but the plan does **not** rely on them as the sole enforcement—EvalContext is authoritative.

---

## 6. Bounds and Clipping

- **CONFIG_DELTAS_BOUND_LOW** and **CONFIG_DELTAS_BOUND_HIGH** define the box for all dimensions.
- **Canonical cluster order** and **list↔dict conversion** are shared (e.g. in `evolution/objective.py` or shared module).
- For optimizers **without native bounds** (e.g. Nelder–Mead, Powell): **clip** proposed `x` into `[CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH]` inside the objective wrapper before calling the actual evaluation (so that EvalContext and SGLang always see feasible points).
- **COBYLA bounds handling (preferred):** Convert box bounds into **inequality constraints** for COBYLA: for each dimension \(i\), \(x_i - \text{low} \ge 0\) and \(\text{high} - x_i \ge 0\). Constraints are thus enforced by the solver, not only by clipping. **Still keep wrapper clipping as a safety net** so that any proposed \(x\) outside the box is clipped before evaluation; the primary enforcement is via constraints.
- **Constraints-only methods (optional / future):** Current optimization is **box-bounded only**. COBYLA and augmented-Lagrangian are included mainly for **future** use with general constraints (or require **CONFIG_USE_CONSTRAINTS=y** to enable). If they remain always available: when the method is constraint-based and **no** extra constraints are configured, **warn** and run with **only box constraints** (COBYLA gets bounds as inequality constraints as above).

---

## 7. Dependencies (Optional Extras)

- **Baseline:** Keep **scipy** as baseline (covers Nelder–Mead, Powell, differential_evolution, DIRECT, etc.).
- **Optional extras:**  
  skopt, pdfo, rbfopt, optuna, smac (+ ConfigSpace), cma (CMA-ES), pyswarms (PSO).  
  Add in plan (and in code): **if the selected method’s import fails, raise a clear error:** e.g. “Install extras for this method: pip install …” (or list the extra name).
- Document in plan: optional dependencies are only required when the corresponding CONFIG_OPTIMIZATION_METHOD is chosen.

---

## 8. Library Usage Details (Fixes)

- **No parallel / no vectorized evaluations:** All optimizers must use **strictly serial** objective evaluation (see §1.2). Force **workers=1**, **n_jobs=1**, **parallel=False**, or equivalent; if a library supports vectorized objective evaluation, **disable it** (our objective cannot be vectorized).
- **Nelder–Mead / Powell (scipy):** Use `scipy.optimize.minimize(..., options={"maxfev": CONFIG_ZERO_ORDER_MAX_EVALS})` so that the solver stops at a finite evaluation count; EvalContext still enforces the true cap and guarantees a returnable best via ctx.best_x/ctx.best_f.
- **Differential evolution (scipy):** Set **workers=1** and avoid multiprocessing. Consider **updating="immediate"** to avoid batch/parallel semantics (e.g. update the best solution after each eval rather than after a full generation). Set **maxiter** and **popsize** so that `popsize * maxiter` is compatible with the global eval cap. **EvalContext still enforces the true cap**—when BudgetExceeded is raised, the driver returns ctx.best_x/ctx.best_f.
- **PSO / CMA-ES (and any population-based method):** If the wrapper can evaluate particles/candidates in parallel, **do not use it**. Evaluate candidates **sequentially** through **ctx.evaluate** (or the single-objective wrapper that goes through EvalContext) so that only one SGLang run is active at a time.
- **DIRECT:** Use `scipy.optimize.direct(func, bounds, maxfun=CONFIG_ZERO_ORDER_MAX_EVALS)` as the **primary budget knob**. Optionally tune `len_tol` / `vol_tol` via CONFIG_DIRECT_LEN_TOL / CONFIG_DIRECT_VOL_TOL if exposed.
- **SMAC:** Use a proper **ConfigurationSpace(seed=...)** with **Float(...)** hyperparameters (one per cluster). Obtain incumbent cost via **smac.validate(incumbent)**; **do not use get_cost**.
- **PDFO:** Use **options={"maxfev": CONFIG_ZERO_ORDER_MAX_EVALS}** and **method="bobyqa"** for bound-constrained problems.

---

## 9. Output, State, and Resume

- **Generic zero_order_state.json** (alongside any GA state) with:
  - **method** name
  - **n_evals_used** (current eval count), **max_evals**, **remaining_evals** (= max_evals - n_evals_used)
  - **best_x**, **best_f**, **best_eval_id** — these refer to the **optimization evaluation set** (per eval_set_mode), or when multi-fidelity is on, to the **confirm-validated best** (best_confirm_x / best_confirm_f). See “best_x meaning” below.
  - **optimizer_seed**, **eval_seed**, **eval_set_mode**, **indices_hash**
  - **history_path** (e.g. zero_order_history.jsonl)
  - **final_full_pool_f** (and optionally shortness_full_pool, correctness_full_pool) — populated when CONFIG_RUN_FINAL_FULL_POOL_EVAL=y; see §9.2.
  - When **CONFIG_USE_MULTI_FIDELITY=y**: **best_search_x**, **best_search_f**; **best_confirm_x**, **best_confirm_f**; **confirm_schedule** (e.g. confirm_every_k_evals, confirm_improvement_eps, search_eval_set_mode, confirm_eval_set_mode, search_minibatch_size, confirm_minibatch_size).
  - bounds, and (optional) method-specific state hook; for **Optuna** / **SMAC** record the path or resume info to their built-in persistence (e.g. Optuna SQLite, SMAC output dir).
- **evaluation_indices.json** (when CONFIG_EVAL_SET_MODE = fixed_minibatch): under `out_dir`, contains the sampled indices and the seed; its hash is **indices_hash**. For **full_pool**, persist resolved pool indices (or hash + source config). For **resampled_minibatch**, log indices or indices_hash per eval in history (see §2.2).
- **deltas_best.json:** saved at end of zero-order run (same as now).
- **processor_best.py:** optionally generated **once at the end** of the run (not per eval).

**Meaning of best_x / best_f:** When multi-fidelity is **off**, best_x and best_f refer to the **optimization evaluation set** (per eval_set_mode). When multi-fidelity is **on**, best_x/best_f are the **confirm-validated best** (best_confirm_x / best_confirm_f). If CONFIG_RUN_FINAL_FULL_POOL_EVAL=y, the **full_pool** score is reported and persisted **separately** (in final_eval.json and zero_order_state.json as final_full_pool_f); **do not overwrite** the optimization-set (or confirm) best_f with the full_pool score.

### 9.1 Resume policy

- **Optuna / SMAC:** Resume via their **native persistence** (SQLite, output directory). Use the paths recorded in zero_order_state.json.
- **Other methods** (skopt, pdfo, direct, differential_evolution, CMA-ES, PSO, Nelder–Mead, Powell, etc.) do not persist full internal algorithm state. Define **resume** as:
  1. Load **zero_order_state.json**; set **x0 = best_x**; set **remaining budget = max_evals - n_evals_used** (or from **remaining_evals**).
  2. Re-create **EvalContext** with identical eval-set config (same **indices_hash**, same pool/minibatch as original run).
  3. Continue optimization with the **same method** if feasible with remaining budget, otherwise **fall back** to a simple local refinement: **recommended default:** pdfo if available, else Powell or Nelder–Mead, using the remaining eval budget.
- **Plan note:** Full internal algorithm-state resume is **not guaranteed** for all methods; **best-effort continuation from best_x** (and remaining_evals) is guaranteed. **best_x/best_f** in state refer to the optimization eval set; when resuming, continue from that best; if final full-pool eval was run, final_full_pool_f is kept separately.

### 9.2 Final evaluation on full_pool (recommended)

- After the optimizer returns, run a **final evaluation** on **full_pool** unless the optimization eval set was already full_pool. The point to evaluate is **best_x** in the single-fidelity case, and **best_confirm_x** when **CONFIG_USE_MULTI_FIDELITY=y** (so the “true best” from confirmation is what gets the final full-pool score).
- **Configurable:** **CONFIG_RUN_FINAL_FULL_POOL_EVAL** (bool, default y). Optional **CONFIG_FINAL_EVAL_SET_MODE** = full_pool | fixed_minibatch for flexibility later.
- **Persist:**
  - **final_eval.json** under `out_dir` containing: `{best_x, f_full_pool, shortness_full_pool, correctness_full_pool, indices_hash_full_pool}` — where **best_x** is the point evaluated (best_x in single-fidelity, **best_confirm_x** when multi-fidelity is enabled).
  - Add to **zero_order_state.json**: **final_full_pool_f** (and optionally shortness_full_pool, correctness_full_pool). Do **not** overwrite best_f (optimization-set or confirm score) with the full_pool score.

---

## 10. CLI Behavior

- **Entry point unchanged:**  
  `python3 -m compressor_2 evolve --config .config`
- **Plan wording:**
  - If **method != deap**, run the **zero-order optimizer path** (EvalContext + dispatcher + chosen optimizer).
  - Save **deltas_best.json** at end.
  - Optionally generate **processor_best.py** only **once at the end** (not per eval).

---

## 11. Python Libraries and Code Structure (Summary)

### 11.1 File layout

- `evolution/objective.py` — shared evaluator; returns fitness to maximize; optional clipping for unbounded methods.
- `evolution/eval_context.py` — EvalContext: cluster_ids, evaluation-set indices, SGLang handle, eval counter, BudgetExceeded (before next eval when at cap), best_x/best_f/best_eval_id, optional cache key (rounded_x_tuple, indices_hash), history JSONL.
- `evolution/optimizers/<method>.py` — each implements `run(ctx, x0, bounds) -> OptimizerResult`; on early termination returns ctx.best_x, ctx.best_f.
- `evolution/optimize_driver.py` — dispatch by cfg.optimization_method; run chosen optimizer; persist deltas_best.json, zero_order_state.json, optional processor_best.py once at end.

### 11.2 Dependencies (requirements / extras)

- **scipy** — baseline (Nelder–Mead, Powell, differential_evolution, direct, etc.).
- **Optional:** skopt, pdfo, rbfopt, optuna, smac, ConfigSpace, cma, pyswarms. On import failure for the selected method, raise: “Install extras for this method: …”.

### 11.3 Snippets (unchanged intent; adapt to EvalContext)

- Optimizers receive **EvalContext** and call a single objective that goes through the context (so budget and history are centralized). The objective callable can be built from `ctx` and `evolution/objective.py` (with optional clipping). Code snippets in the original plan (§4.2–4.8) should be adapted so that:
  - The minimizer receives a wrapper that (1) optionally clips x, (2) calls ctx.evaluate(x) or equivalent that increments counter and raises BudgetExceeded, (3) returns -f.
  - DIRECT uses `maxfun=ctx.max_evals` (or cfg.zero_order_max_evals).
  - SMAC uses ConfigurationSpace with Float and smac.validate(incumbent).
  - PDFO uses method="bobyqa" and options={"maxfev": ...}.

---

## 12. Implementation Checklist (for implementer)

1. **Kconfig:** Add CONFIG_OPTIMIZATION_METHOD (full choice list), CONFIG_EVAL_SET_MODE, CONFIG_EVAL_MINIBATCH_SIZE, CONFIG_EVAL_SEED, CONFIG_OPTIMIZER_SEED, CONFIG_ZERO_ORDER_MAX_EVALS, CONFIG_DELTAS_BOUND_LOW/HIGH, CONFIG_EVAL_TIMEOUT_S, CONFIG_EVAL_MAX_RETRIES, CONFIG_EVAL_FAILURE_FITNESS, CONFIG_CACHE_ROUND_DECIMALS or CONFIG_CACHE_EPS, **CONFIG_ENABLE_CACHE** (optional), **CONFIG_EVAL_DETERMINISTIC**, **CONFIG_INFERENCE_SEED** or CONFIG_SGLANG_SEED, CONFIG_TEMPERATURE/CONFIG_TOP_P if not present, **CONFIG_RUN_FINAL_FULL_POOL_EVAL** (bool, default y), optional **CONFIG_FINAL_EVAL_SET_MODE**, CONFIG_USE_CONSTRAINTS (optional), gradient-estimate knobs (CONFIG_ZO_*), and method-specific options. Parser and validation (GA only when method=deap). Optionally error or warn when a surrogate method is used with resampled_minibatch. **Method-specific validation:** SPSA/2-pt/Kiefer–Wolfowitz: max_evals >= 2, derived T >= 1; CONFIG_ZO_NUM_DIRECTIONS >= 1 when applicable; CONFIG_CACHE_ROUND_DECIMALS in 0–12 or CONFIG_CACHE_EPS > 0; fixed_minibatch/resampled_minibatch: CONFIG_EVAL_MINIBATCH_SIZE <= len(pool_indices).
2. **EvalContext:** Implement `evolution/eval_context.py` with cluster_ids, eval-set indices (full_pool / fixed_minibatch / resampled_minibatch), SGLang handle, n_evals and BudgetExceeded (triggered before starting next eval when n_evals == max_evals); **each attempt consumes budget** (including retries and failures). best_x/best_f/best_eval_id updated **only on successful evals**. **Single evaluation at a time:** no parallel or vectorized objective calls; all evals serialized through context. Failure handling: timeout (CONFIG_EVAL_TIMEOUT_S), retries (CONFIG_EVAL_MAX_RETRIES), CONFIG_EVAL_FAILURE_FITNESS on total failure; log failures in JSONL with {eval_id, x, status, error, attempt, indices_hash, timestamp}. **Caching:** only when evaluation is deterministic and eval_set_mode != resampled_minibatch (or CONFIG_ENABLE_CACHE); cache key (rounded_x_tuple, indices_hash) using CONFIG_CACHE_ROUND_DECIMALS or CONFIG_CACHE_EPS; cache disabled by default in resampled_minibatch unless keyed by per-eval indices. **History JSONL:** include decoding settings and inference seed (or hash) per line for audit; successes and failures.
3. **Objective:** Implement `evolution/objective.py` with shared canonical order and list↔dict conversion; objective wrapper accepts x as list or numpy array, converts to list of floats, clips into bounds; evaluate_objective; wrapper for minimizers with optional clipping for unbounded methods. **x0:** clip into bounds once at init for all methods and log clipped x0.
4. **Optimizers:** Implement each method under `evolution/optimizers/` with signature `run(ctx, x0, bounds) -> OptimizerResult`; on any early termination (BudgetExceeded or library abort) return ctx.best_x, ctx.best_f; **force serial evaluation** (workers=1, n_jobs=1, no vectorized objective)—see §8. Pass library-specific maxiter/maxfun/n_trials where applicable (Nelder–Mead/Powell: options={"maxfev": ...}; differential evolution: workers=1, updating="immediate", maxiter/popsize; PSO/CMA-ES: evaluate candidates sequentially via ctx.evaluate; EvalContext still enforces true cap).
5. **Dispatcher:** Implement `evolution/optimize_driver.py` to dispatch by cfg.optimization_method, run optimizer, save deltas_best.json, **zero_order_state.json** (with n_evals_used, max_evals, remaining_evals, best_x, best_f, best_eval_id, optimizer_seed, eval_seed, eval_set_mode, indices_hash, history_path, final_full_pool_f when applicable—see §9), **evaluation_indices.json** when fixed_minibatch (and persist pool/hash for full_pool), **final_eval.json** when CONFIG_RUN_FINAL_FULL_POOL_EVAL=y (best_x, f_full_pool, shortness_full_pool, correctness_full_pool, indices_hash_full_pool—see §9.2), and optionally processor_best.py once at end. **Final stage:** after optimizer returns, run final eval of best_x on full_pool unless eval_set_mode=full_pool; persist final_eval.json and final_full_pool_f in state; do not overwrite best_f. **Resume:** support loading zero_order_state.json and continuing (Optuna/SMAC via native persistence; others: x0=best_x, remaining_evals, same or fallback method—see §9.1). **Temp artifacts:** use eval_id in filenames and keep only last N (see §1.2).
6. **CLI:** In `evolve` entry point, if method != deap run zero-order path; save deltas_best.json at end; optionally generate processor_best.py only once at end.
7. **Dependencies:** scipy baseline; optional extras with clear “install extras: …” on import failure.
8. **Tests:** Optional unit test for objective with mock; optional integration test with one method and tiny pool.

---

## 13. Summary

- **Objective:** Maximize `f = lambda_shortness * shortness_score + lambda_correctness * correctness_ratio`; same for all methods.
- **Scope:** All methods from zero_order_opt.md and extended list (grid, random, coordinate, Hooke–Jeeves, Nelder–Mead, Powell, skopt, pdfo, rbfopt, DIRECT, CMA-ES, differential evolution, PSO, SPSA, random-direction 1-pt/2-pt, Kiefer–Wolfowitz, COBYLA, augmented-Lagrangian, optuna, smac, hybrid), selectable via Kconfig.
- **Unified interface:** EvalContext (cluster_ids, eval-set indices, SGLang, budget, cache, history); optimizers implement `run(ctx, x0, bounds) -> OptimizerResult`; optimize_driver dispatches and persists.
- **Eval-set mode:** full_pool | fixed_minibatch | resampled_minibatch (CONFIG_EVAL_SET_MODE, CONFIG_EVAL_MINIBATCH_SIZE, CONFIG_EVAL_SEED). Surrogate methods (skopt, SMAC, TPE, RBFOpt, DIRECT) not recommended with resampled_minibatch; optionally error or warn.
- **Seeds:** CONFIG_OPTIMIZER_SEED (stochastic optimizers); CONFIG_EVAL_SEED (fixed_minibatch sampling only).
- **Budget:** CONFIG_ZERO_ORDER_MAX_EVALS enforced in EvalContext; next eval triggers BudgetExceeded before SGLang when cap reached; EvalContext keeps best_x/best_f/best_eval_id; drivers return ctx.best_x/ctx.best_f on early termination.
- **Bounds:** CONFIG_DELTAS_BOUND_LOW/HIGH; clip in wrapper for methods without native bounds; COBYLA: prefer box bounds as inequality constraints, clipping as safety net; canonical order and list↔dict shared.
- **Failure policy:** CONFIG_EVAL_TIMEOUT_S, CONFIG_EVAL_MAX_RETRIES, CONFIG_EVAL_FAILURE_FITNESS; each attempt consumes budget; best_x/best_f only on success; failures logged in JSONL (eval_id, x, status, error, attempt, indices_hash, timestamp).
- **Reproducibility:** evaluation_indices.json (fixed_minibatch: indices + seed; full_pool: resolved pool or hash; resampled: indices/hash per eval in history). Cache precision: CONFIG_CACHE_ROUND_DECIMALS or CONFIG_CACHE_EPS; rounded_x_tuple for cache and duplicate detection. **Determinism and cache:** cache only when evaluation is deterministic (CONFIG_EVAL_DETERMINISTIC / fixed inference seed) and eval_set_mode != resampled; if stochastic, disable cache by default or CONFIG_ENABLE_CACHE. History logs decoding settings and inference seed (or hash). Cache disabled by default in resampled_minibatch unless keyed by per-eval indices.
- **Evaluation determinism:** Prefer deterministic decoding (temperature=0, top_p=1, CONFIG_INFERENCE_SEED); CONFIG_EVAL_DETERMINISTIC, CONFIG_TEMPERATURE/CONFIG_TOP_P if not elsewhere (§1.2).
- **best_x / best_f meaning:** Refer to optimization eval set; if RUN_FINAL_FULL_POOL_EVAL=y, full_pool score persisted separately (final_eval.json, final_full_pool_f); do not overwrite optimization-set best_f (§9, §9.1, §9.2).
- **Final full-pool eval:** CONFIG_RUN_FINAL_FULL_POOL_EVAL (default y); after optimizer returns, eval best_x on full_pool unless already full_pool; persist final_eval.json and final_full_pool_f in zero_order_state.json (§9.2).
- **Objective contract:** wrapper accepts list or numpy array, converts to list of floats, clips; x0 clipped once at init and logged. Constraints: box-bounded only; COBYLA/augmented-Lagrangian optional/future (CONFIG_USE_CONSTRAINTS or warn when no extra constraints).
- **No parallel evaluations:** Single evaluation at a time; workers=1 / n_jobs=1 / no vectorized objective; DE with updating="immediate"; PSO/CMA-ES evaluate candidates sequentially via ctx.evaluate (§8).
- **Resume:** zero_order_state.json includes n_evals_used, remaining_evals, best_x/best_f/best_eval_id (optimization set), seeds, indices_hash, history_path, final_full_pool_f when applicable; Optuna/SMAC resume natively; others resume from best_x with remaining budget (same method or fallback pdfo/Powell/Nelder–Mead); best-effort continuation guaranteed (§9.1).
- **Validation:** SPSA/2-pt/KW: max_evals >= 2, T >= 1; CONFIG_ZO_NUM_DIRECTIONS >= 1; cache decimals 0–12 or EPS > 0; fixed/resampled minibatch: minibatch_size <= len(pool) (§2.4).
- **Temp artifacts:** eval_id in filenames; keep only last N to limit disk use (§1.2).
- **Output:** deltas_best.json, zero_order_state.json (method, n_evals_used, max_evals, remaining_evals, best_x/best_f/best_eval_id, seeds, eval_set_mode, indices_hash, history path, final_full_pool_f), evaluation_indices.json when applicable, **final_eval.json** when CONFIG_RUN_FINAL_FULL_POOL_EVAL=y; processor_best.py once at end.
- **CLI:** `compressor_2 evolve --config .config`; if method != deap run zero-order path; save deltas_best.json at end; optionally generate processor_best.py only once at end.
