# Plan: Coordinate-Then-Random-Direction Zero-Order Method

This document is an **implementation plan** for adding a new zero-order optimization method **`coordinate_then_random_direction`** to the compressor_2 repo. The method integrates with the **existing** zero-order framework; it does **not** re-invent evaluation, GPU pinning, or SGLang lifecycle.

---

## 0. Context and Assumptions (must respect)

The following are **fixed**; the new method must conform.

- **Evaluation pipeline is shared and already implemented:** `evolution/objective.evaluate_x` is the only allowed evaluator. The new optimizer **must not** call `evaluate_x` directly.
- **Zero-order must evaluate only through EvalContext:** Use `ctx.evaluate(x, next_x=...)` or **`ctx.evaluate_batch(xs)`** so evaluations benefit from prefetch and use the shared pipeline.
- **Exactly 2 visible GPUs required.** Use `validate_2_gpus()` from `evolution/gpu_utils.py` and the same pinning rules:
  - **generate-processor** on **GPU-A** (`cuda:0` in parent), `env=os.environ`, no CVD override.
  - **SGLang** started **once** and kept running, pinned to **GPU-B** via CVD shim only (no per-eval SGLang restart).
- **Eval set is ALWAYS fixed minibatch:** CONFIG_EVAL_MINIBATCH_SIZE + CONFIG_EVAL_SEED; persist `evaluation_indices.json` and `indices_hash`. No full_pool or resampled_minibatch.
- **Deterministic evaluation recommended:** CONFIG_EVAL_DETERMINISTIC=y (temperature=0, top_p=1, fixed seed). If stochastic, caching is disabled by default.
- **Budget semantics:** CONFIG_ZERO_ORDER_MAX_EVALS is the global hard cap; each evaluation attempt consumes budget; **BudgetExceeded** is raised before starting the next eval when budget would be exceeded.
- **Failure policy:** CONFIG_EVAL_TIMEOUT_S, CONFIG_EVAL_MAX_RETRIES, CONFIG_EVAL_FAILURE_FITNESS; failures logged to history JSONL; best_x / best_f updated only on successful evaluations.

See `docs/zero_order_opt_plan.md` for full pipeline and terminology (EvalContext, prefetch, 2-GPU enforcement, etc.).

---

## 1. Overview

### 1.1 Algorithm name and goal

- **Method name:** `coordinate_then_random_direction`
- **Goal:** Maximize `f(x)` over `x in R^d` (d = number of clusters, canonical order). Maintain current point `x_t`; use coordinate search (standard-basis directions) for the first **CONFIG_COORD_RD_COORD_K** iterations (phase 1), then switch to random-direction steps (phase 2) until budget is exhausted. Phase 1 and phase 2 use **separate** hyperparameters (alpha0, alpha_min, shrink, improvement_eps, etc.); see §2.2.

### 1.2 High-level structure and direction sampling

**Direction sampling is phase-specific and explicit:**

- **Phase 1 — Coordinate search (first K iterations):** Sample directions from the **standard basis**: choose coordinate index `i` and use direction **e_i** (unit vector along dimension i).  
  - **Selection policy:** If `CONFIG_COORD_RD_COORD_NUM_COORDS_PER_ITER == d` or **0 (all)**: iterate over all basis directions (optionally shuffled per CONFIG_COORD_RD_COORD_SHUFFLE_EACH_ITER).  
  - If `NUM_COORDS_PER_ITER < d`: sample a **subset** of indices each iteration using the RNG seeded by CONFIG_OPTIMIZER_SEED; with or without replacement per CONFIG_COORD_RD_COORD_SAMPLE_WITH_REPLACEMENT.  
  For each chosen coordinate `i`, evaluate `x + alpha * e_i` and `x - alpha * e_i` via **`ctx.evaluate_batch([x_plus, x_minus])`**. Compute directional derivative estimate `g_i_hat = (f_plus - f_minus) / (2*alpha)`. Accept an improving move (best among tested, subject to CONFIG_COORD_RD_COORD_IMPROVEMENT_EPS). If no improvement during the iteration, shrink alpha by CONFIG_COORD_RD_COORD_SHRINK. Clip all proposals to [CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH]. Optional cap CONFIG_COORD_RD_COORD_MAX_COORDS_TOTAL (if > 0) and global max_evals are both enforced.
- **Phase 2 — Random direction:** Sample a **random unit direction** u from the configured distribution (**gaussian_unit** or **rademacher_unit** per CONFIG_COORD_RD_RAND_DIR_DIST). Evaluate `x + alpha*u` and `x - alpha*u` via **`ctx.evaluate_batch([x_plus, x_minus])`**. Compute `g_u_hat = (f_plus - f_minus) / (2*alpha)`. Accept best improving move (CONFIG_COORD_RD_RAND_IMPROVEMENT_EPS). Shrink alpha when no improvement (CONFIG_COORD_RD_RAND_SHRINK). Repeat for CONFIG_COORD_RD_RAND_DIRS_PER_ITER directions per iteration until budget (and optionally CONFIG_COORD_RD_RAND_MAX_DIR_PAIRS_TOTAL) exhausted.

### 1.3 Derivative magnitude tracking (required)

- **Standard-basis derivative magnitude:** Updated **only in phase 1**. Each time a ± pair along coordinate `i` (direction e_i) is evaluated, compute `g_i_hat = (f(x+alpha e_i) - f(x-alpha e_i)) / (2*alpha)`. Maintain running mean of absolute values: **mean_abs_grad_basis** = mean over all computed `|g_i_hat|` so far (across iterations and coordinates). Count: **n_basis_pairs**.
- **Random-direction derivative magnitude:** Updated **only in phase 2**. Each time a ± pair along random direction `u` is evaluated, compute `g_u_hat = (f(x+alpha u) - f(x-alpha u)) / (2*alpha)`. Maintain **mean_abs_grad_rand** = mean over all computed `|g_u_hat|` so far. Count: **n_rand_pairs**.
- If a mean is undefined (count 0), log it as `null` in JSON outputs.

---

## 2. Kconfig Additions

### 2.1 Optimization method choice

- Add to **CONFIG_OPTIMIZATION_METHOD** (choice): **`coordinate_then_random_direction`**.
- When this method is selected, the loader uses `validate_ga=False` and loads zero-order options plus the method-specific options below (same as other zero-order methods).

### 2.2 Method-specific symbols — phase-specific (no shared alpha/step/shrink)

**Two separate sets.** Do **not** reuse one alpha/step/shrink for both phases. All method-specific symbols live under a submenu when method is `coordinate_then_random_direction`.

#### Phase 1 — Coordinate / standard-basis

| Symbol | Type | Default | Description |
|--------|------|---------|-------------|
| CONFIG_COORD_RD_COORD_K | int | e.g. 10 | Number of coordinate-search **iterations** in phase 1 (rename from CONFIG_COORD_RD_K). Must be > 0. |
| CONFIG_COORD_RD_COORD_ALPHA0 | float | e.g. 0.1 | Initial step size alpha for phase 1. Must be > 0. |
| CONFIG_COORD_RD_COORD_ALPHA_MIN | float | e.g. 1e-6 | Minimum step size for phase 1; do not shrink below this. Must be > 0, alpha_min <= alpha0. |
| CONFIG_COORD_RD_COORD_SHRINK | float | e.g. 0.5 | Shrink factor in (0, 1). New alpha = alpha * SHRINK when no improvement in a coordinate iteration. |
| CONFIG_COORD_RD_COORD_NUM_COORDS_PER_ITER | int | 0 | Coordinates per iteration: 1..d, or **0 = all d**. Validated at runtime: in [1, d] or 0. |
| CONFIG_COORD_RD_COORD_OPPORTUNISTIC | bool | n | If y: stop at first improving coordinate move within an iteration. If n: exhaustive over chosen coordinates, then take best improving move. |
| CONFIG_COORD_RD_COORD_IMPROVEMENT_EPS | float | 0.0 | Minimum improvement to accept a move in phase 1: accept only if new_f > current_f + EPS. >= 0. |
| CONFIG_COORD_RD_COORD_SAMPLE_WITH_REPLACEMENT | bool | n | **Optional.** If y: when NUM_COORDS_PER_ITER < d, sample coordinate indices with replacement; if n: without replacement. |
| CONFIG_COORD_RD_COORD_SHUFFLE_EACH_ITER | bool | y or n | **Optional.** If y: shuffle coordinate indices each iteration (when using subset or all). |
| CONFIG_COORD_RD_COORD_MAX_COORDS_TOTAL | int | 0 | **Optional.** Hard cap on coordinate ± pairs in phase 1; 0 = disabled. If set > 0, **both** this cap and global max_evals are enforced (stop when either is hit). |

#### Phase 2 — Random-direction

| Symbol | Type | Default | Description |
|--------|------|---------|-------------|
| CONFIG_COORD_RD_RAND_ALPHA0 | float | e.g. 0.1 | Initial step size alpha for phase 2. Must be > 0. **At phase switch, alpha is reset to this value** (see §3.3). |
| CONFIG_COORD_RD_RAND_ALPHA_MIN | float | e.g. 1e-6 | Minimum step size for phase 2. Must be > 0, alpha_min <= alpha0. |
| CONFIG_COORD_RD_RAND_SHRINK | float | e.g. 0.5 | Shrink factor in (0, 1). New alpha = alpha * SHRINK when no improvement in a random-direction iteration. |
| CONFIG_COORD_RD_RAND_DIR_DIST | choice | gaussian_unit | How to sample unit direction u: **gaussian_unit** (z ~ N(0,I), u = z/||z||) or **rademacher_unit** (components ±1 uniformly, then normalize). (Renamed from CONFIG_COORD_RD_RANDOM_DIR_DIST.) |
| CONFIG_COORD_RD_RAND_DIRS_PER_ITER | int | 1 | Random directions per iteration. Must be >= 1. (Renamed from CONFIG_COORD_RD_RANDOM_DIRS_PER_ITER.) |
| CONFIG_COORD_RD_RAND_IMPROVEMENT_EPS | float | 0.0 | Minimum improvement to accept a move in phase 2. >= 0. |
| CONFIG_COORD_RD_RAND_USE_CURRENT_X_FOR_NEXT_DIR | bool | y | **Optional.** If y: use current best x within the iteration as starting point for the next direction (default). If n: use x at start of iteration for all directions in that iteration. |
| CONFIG_COORD_RD_RAND_MAX_DIR_PAIRS_TOTAL | int | 0 | **Optional.** Hard cap on random-direction ± pairs in phase 2; 0 = disabled. If set > 0, both this cap and global max_evals are enforced. |

- **RNG seed:** **CONFIG_OPTIMIZER_SEED** is the sole RNG seed for **both** phases unless separate per-phase seeds are added; if separate seeds are added, document precedence and determinism.
- **Reuse** global config for bounds (CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH), CONFIG_OPTIMIZER_SEED, determinism, and all evaluation/budget options.

---

## 3. Optimizer Implementation Details

### 3.1 File and contract

- **File:** `evolution/optimizers/coordinate_then_random_direction.py`
- **Contract:** `run(ctx: EvalContext, x0: List[float], bounds: Tuple[float, float], cfg: EvolutionConfig) -> OptimizerResult`
- **Return value:** The optimizer must return **OptimizerResult(..., method_summary={...})** where **method_summary** is a dict with at least **mean_abs_grad_basis**, **mean_abs_grad_rand**, **n_basis_pairs**, **n_rand_pairs** (final values) so that optimize_driver can write final_eval.json even if ctx.get_method_state is missing (see §4.3). The OptimizerResult dataclass must support an optional **method_summary** (or **extras**) field.
- **Evaluation:** Every ± pair **must** be evaluated via **`ctx.evaluate_batch([x_plus, x_minus])`**. Do **not** call `evaluate_x` or `ctx.evaluate` for these pairs (so prefetch is used). Before each ± pair, ensure **remaining budget >= 2** (see §3.10). After each pair, update running stats and call **ctx.set_extra_log_fields(...)** and **ctx.set_method_state("coord_rd", ...)** (see §4.1, §4.2, §4.2.1). The **current** running stats must be attached to the standard EvalContext JSONL eval records (see §4.1) via the EvalContext extra-fields API.

### 3.2 Iteration unit (precise definition)

- **Coordinate phase:** One **iteration** = one pass over the chosen coordinates for the current x and alpha. If CONFIG_COORD_RD_COORD_OPPORTUNISTIC=y, the iteration may stop early on first improvement; otherwise it tests all CONFIG_COORD_RD_COORD_NUM_COORDS_PER_ITER (or d when 0) coordinates, then applies the best improving move (if any). After the iteration: if no improvement, set `alpha = max(alpha_min, alpha * CONFIG_COORD_RD_COORD_SHRINK)`.
- **Random-direction phase:** One **iteration** = trying CONFIG_COORD_RD_RAND_DIRS_PER_ITER random directions. For each direction, evaluate the ± pair, update x if improvement (best of x_plus, x_minus, and current x; CONFIG_COORD_RD_RAND_USE_CURRENT_X_FOR_NEXT_DIR controls whether “current x” is the start-of-iteration x or the best so far within the iteration). After the iteration: if no improvement over the iteration, set `alpha = max(alpha_min, alpha * CONFIG_COORD_RD_RAND_SHRINK)`.

### 3.3 Alpha update rule and phase boundary

- **Shrink (per phase):** When no improvement is found during an iteration (coordinate or random), set `alpha = max(alpha_min, alpha * shrink)` using the **active phase’s** shrink (CONFIG_COORD_RD_COORD_SHRINK or CONFIG_COORD_RD_RAND_SHRINK) and alpha_min (COORD_ALPHA_MIN or RAND_ALPHA_MIN).
- **Phase boundary (phase 1 → phase 2):** When switching from coordinate to random-direction phase, **reset alpha to CONFIG_COORD_RD_RAND_ALPHA0** (do not carry over the current coordinate-phase alpha). This is the defined rule.
- **Iteration counters:** Maintain two separate counters: **coord_iter** = 0 .. K−1 during phase 1; **rand_iter** = 0, 1, 2, … during phase 2 until budget (and optional RAND_MAX_DIR_PAIRS_TOTAL) exhausted. Use these for logging and state (current_iteration can be reported as coord_iter in phase 1 and rand_iter in phase 2, or both stored; see §4).
- **Do not increase alpha** in this plan (no expansion step). Alpha only shrinks or stays the same when there is an improvement (alpha unchanged when a move is accepted).

### 3.4 Move acceptance rule

- **Candidate set:** For coordinate step along `i`: candidates are `x`, `x + alpha*e_i` (clipped), `x - alpha*e_i` (clipped). For random step: candidates are `x`, `x + alpha*u` (clipped), `x - alpha*u` (clipped). Current `f(x)` is already known from the previous step (or from initial eval of x0).
- **Accept:** Choose the candidate with the largest f. **Update x** if that best candidate has `f > f_current + improvement_eps`, where **improvement_eps** is CONFIG_COORD_RD_COORD_IMPROVEMENT_EPS in phase 1 and CONFIG_COORD_RD_RAND_IMPROVEMENT_EPS in phase 2. Otherwise keep x and shrink alpha (if no improvement in this iteration).

### 3.5 Bound handling

- All proposals `x_plus`, `x_minus` must be **clipped** component-wise to `[CONFIG_DELTAS_BOUND_LOW, CONFIG_DELTAS_BOUND_HIGH]` before calling `ctx.evaluate_batch`. Use the same clipping helper as in SPSA/random_direction_2pt (e.g. per-component min/max).

### 3.6 RNG

- Use **CONFIG_OPTIMIZER_SEED** to seed a single `random.Random` (or numpy Generator) for **both** phases. Use it for: (1) **Phase 1:** coordinate index selection/ordering when CONFIG_COORD_RD_COORD_NUM_COORDS_PER_ITER < d (e.g. random permutation or sample with/without replacement per CONFIG_COORD_RD_COORD_SAMPLE_WITH_REPLACEMENT), and optionally shuffle when CONFIG_COORD_RD_COORD_SHUFFLE_EACH_ITER=y; (2) **Phase 2:** sampling random unit direction u per CONFIG_COORD_RD_RAND_DIR_DIST (gaussian_unit or rademacher_unit). Document that both phases share this RNG; if separate per-phase seeds are ever added, document precedence and determinism.

### 3.7 Pseudocode (coordinate phase, one iteration)

- **Direction sampling (phase 1):** Sample directions from the **standard basis**: for each chosen coordinate index `i`, direction is **e_i**. If NUM_COORDS_PER_ITER is 0 or d: iterate over all indices [0..d-1] (optionally shuffled). If NUM_COORDS_PER_ITER < d: sample a subset of size NUM_COORDS_PER_ITER using RNG (with or without replacement per config).

```
coordinate_iteration(x, alpha, bounds, cfg, ctx):
  lo, hi = bounds
  d = len(x)
  n_per = cfg.coord_rd_coord_num_coords_per_iter
  coords = (indices 0..d-1, optionally shuffled) if n_per == 0 or n_per >= d
           else sample n_per indices from 0..d-1 (RNG; with/without replacement per cfg.coord_rd_coord_sample_with_replacement)
  best_f = current f(x)
  best_x = x
  for i in coords:
    if ctx.max_evals - ctx.n_evals < 2:
      raise BudgetExceeded
    if cfg.coord_rd_coord_max_coords_total > 0 and n_coord_pairs_done >= cfg.coord_rd_coord_max_coords_total:
      break
    e_i = unit vector along i   # standard basis direction
    x_plus = clip(x + alpha * e_i, lo, hi)
    x_minus = clip(x - alpha * e_i, lo, hi)
    f_plus, f_minus = ctx.evaluate_batch([x_plus, x_minus])
    g_i_hat = (f_plus - f_minus) / (2 * alpha)
    update mean_abs_grad_basis with |g_i_hat|; n_basis_pairs += 1
    for cand, f_cand in [(x_plus, f_plus), (x_minus, f_minus)]:
      if f_cand > best_f + cfg.coord_rd_coord_improvement_eps:
        best_f, best_x = f_cand, cand
    if cfg.coord_rd_coord_opportunistic and best_x != x:
      break
  if best_x != x:
    x = best_x
  else:
    alpha = max(cfg.coord_rd_coord_alpha_min, alpha * cfg.coord_rd_coord_shrink)
  return x, alpha
```

### 3.8 Pseudocode (random-direction phase, one iteration)

- **Direction sampling (phase 2):** Sample a **random unit direction** u from the configured distribution: **gaussian_unit** (z ~ N(0,I), u = z/||z||) or **rademacher_unit** (each component ±1 uniformly, then u = v/||v||). Use RNG seeded by CONFIG_OPTIMIZER_SEED.

```
random_direction_iteration(x, alpha, bounds, cfg, ctx):
  lo, hi = bounds
  d = len(x)
  best_f = current f(x)
  best_x = x
  x_start = x
  for _ in range(cfg.coord_rd_rand_dirs_per_iter):
    if ctx.max_evals - ctx.n_evals < 2:
      raise BudgetExceeded
    if cfg.coord_rd_rand_max_dir_pairs_total > 0 and n_rand_pairs_done >= cfg.coord_rd_rand_max_dir_pairs_total:
      break
    u = sample_unit_direction(d, cfg.coord_rd_rand_dir_dist, rng)   # gaussian_unit or rademacher_unit
    x_plus = clip(x + alpha * u, lo, hi)
    x_minus = clip(x - alpha * u, lo, hi)
    f_plus, f_minus = ctx.evaluate_batch([x_plus, x_minus])
    g_u_hat = (f_plus - f_minus) / (2 * alpha)
    update mean_abs_grad_rand with |g_u_hat|; n_rand_pairs += 1
    for cand, f_cand in [(x_plus, f_plus), (x_minus, f_minus)]:
      if f_cand > best_f + cfg.coord_rd_rand_improvement_eps:
        best_f, best_x = f_cand, cand
    if cfg.coord_rd_rand_use_current_x_for_next_dir:
      x = best_x
  if best_x != x_start:
    x = best_x
  else:
    alpha = max(cfg.coord_rd_rand_alpha_min, alpha * cfg.coord_rd_rand_shrink)
  return x, alpha
```

### 3.9 Initial evaluation of x0

- Before the first coordinate iteration, evaluate x0 once to get f(x0). Use **`ctx.evaluate(x0)`** (or one call to evaluate_batch with a single point if the API allows). This consumes one eval. Then run **CONFIG_COORD_RD_COORD_K** coordinate iterations (phase 1), then switch to phase 2 and run random-direction iterations until budget (and optional RAND_MAX_DIR_PAIRS_TOTAL) exhausted. At the phase switch, **reset alpha to CONFIG_COORD_RD_RAND_ALPHA0** (see §3.3).

### 3.10 Budget safety inside an iteration

- **Rule:** Before calling **`ctx.evaluate_batch([x_plus, x_minus])`**, the optimizer **must** check that **remaining_evals >= 2** (because each ± pair consumes two evaluation attempts, including any retries/failures counted by EvalContext). Compute remaining as `ctx.max_evals - ctx.n_evals`.
- If **remaining_evals < 2**, do **not** start the pair. Instead, stop cleanly: trigger the normal BudgetExceeded path (return OptimizerResult with ctx.best_x, ctx.best_f, etc.) so that the driver's `finally` block persists state. The persisted state must include the latest alpha, phase, iteration index, and derivative means/counts (see §4.2), so that resume or post-hoc analysis has a consistent snapshot.
- This avoids entering evaluate_batch with insufficient budget and ensures state is always persisted with valid coord_rd fields when the run ends due to budget.

---

## 4. Stats Tracking, Logging, and Plotting

### 4.1 zero_order_history.jsonl — add fields to standard eval records (required)

- **Requirement:** The **current** running stats must be included as **fields on the standard EvalContext JSONL evaluation records**. Each eval record written by EvalContext (for both `evaluate()` and `evaluate_batch()`) must carry the latest stats when set.

- **Mechanism (single choice):** Add the following EvalContext API:
  - **`ctx.set_extra_log_fields(d: dict)`** — set the dict that will be merged into every subsequent eval JSONL record.
  - **`ctx.get_extra_log_fields() -> dict`** — return the current extras dict (for reading/merging elsewhere).
  - EvalContext **merges** the current extras dict into **every** eval record it writes in `_log_history` (for both single-point evals and each record from `evaluate_batch`). Keys in the extras dict are added to the record; missing keys are omitted. Clearing or updating is the optimizer’s responsibility.

- **Exact keys for this method:** The optimizer must set (at least) these keys when calling `ctx.set_extra_log_fields(...)`:
  - **mean_abs_grad_basis**, **mean_abs_grad_rand** (float or null)
  - **n_basis_pairs**, **n_rand_pairs** (int)
  - **current_alpha** (float) — the **active phase’s** alpha (coord phase: coordinate alpha; rand phase: random alpha)
  - **coord_alpha**, **rand_alpha** (float, optional) — both phase alphas for visibility when desired
  - **current_phase** ("coordinate" | "random_direction"), **current_iteration** (int; coord_iter in phase 1, rand_iter in phase 2)

- **When the optimizer must call `ctx.set_extra_log_fields(...)`:**
  - **After updating the running means for each ± pair** — so that the next eval record(s) written by EvalContext (the two lines for that pair, or at least the second) include the updated stats.
  - **At phase switch** — when switching from coordinate to random_direction, set extras (and method_state) so that subsequent records and state checkpoints reflect the new phase.

### 4.2 zero_order_state.json — data path (no magic)

- **Existing fields** (method, n_evals_used, max_evals, best_x, best_f, best_eval_id, optimizer_seed, eval_seed, indices_hash, history_path) are written as today.

- **Additional fields for this method only** (same keys as in §4.1 / §4.3 where applicable):  
  last_mean_abs_grad_basis, last_mean_abs_grad_rand, n_basis_pairs, n_rand_pairs;  
  **coord_alpha_current**, **rand_alpha_current** (float; current alpha in each phase);  
  current_alpha (active phase’s alpha), current_phase, current_iteration (and optionally coord_iter, rand_iter);  
  **coord hyperparams** and **rand hyperparams** (or at minimum the Kconfig values that define them for reproducibility: e.g. coord_k, coord_alpha0, coord_alpha_min, coord_shrink, coord_num_coords_per_iter, coord_opportunistic, coord_improvement_eps; rand_alpha0, rand_alpha_min, rand_shrink, rand_dir_dist, rand_dirs_per_iter, rand_improvement_eps, and optional caps).

- **How optimize_driver obtains them:** Use **method state** on EvalContext:
  - Add to EvalContext: **`ctx.set_method_state(name: str, state: dict)`** and **`ctx.get_method_state(name: str) -> dict | None`**.  
  - The optimizer calls **`ctx.set_method_state("coord_rd", {...})`** with a dict containing the current stats (mean_abs_grad_basis, mean_abs_grad_rand, n_basis_pairs, n_rand_pairs), **coord_alpha_current**, **rand_alpha_current**, current_alpha, current_phase, current_iteration (and optionally coord_iter, rand_iter), and optionally the phase hyperparams or Kconfig names for reproducibility. It must do this **at least once per iteration** (and at phase switch and at end) so that the driver can read the latest values.
  - **optimize_driver**, in the **finally** block when persisting `zero_order_state.json`, **if** `cfg.optimization_method == "coordinate_then_random_direction"`, calls **`ctx.get_method_state("coord_rd")`**. If the returned dict is non-None, merge its contents into the state dict (flatten the keys as the field names already specified, e.g. last_mean_abs_grad_basis, etc.) before writing `zero_order_state.json`. No magic or method-specific logic outside this contract.


### 4.2.1 State checkpoint frequency (report during the run)

- At the end of **each** iteration (coordinate or random_direction), the optimizer **must** call **`ctx.set_extra_log_fields(...)`** and **`ctx.set_method_state("coord_rd", ...)`** with the latest stats (and phase, iteration, alpha). Then: (1) history eval lines written afterward carry the latest stats, and (2) if the run is interrupted (SIGINT or exception), the next periodic checkpoint or SIGINT save will write the latest values from **ctx.get_method_state("coord_rd")**.
- If optimize_driver already checkpoints state after each eval (or on a timer), reading **ctx.get_method_state("coord_rd")** in that path is sufficient — the fields will be present because the optimizer updates them at least once per iteration (and after each ± pair for extras). If the driver checkpoints **less often** than once per iteration, the optimizer must still update method_state and extra_log_fields **at least once per iteration** so that whenever the driver does persist, it has the latest; optionally require that for this method the driver checkpoints at least once per iteration if it does not already persist frequently enough.

### 4.3 Final summary (final_eval.json or separate summary)

- Add to **final_eval.json** (or a separate summary JSON when method is coordinate_then_random_direction):  
  final_mean_abs_grad_basis, final_mean_abs_grad_rand, final_n_basis_pairs, final_n_rand_pairs.

- **How optimize_driver obtains them (no magic):**
  - **Primary:** When writing final_eval.json, if method is coordinate_then_random_direction, call **`ctx.get_method_state("coord_rd")`** and write the final means and counts from that dict (with the final_* key names above). The optimizer must have called `ctx.set_method_state("coord_rd", ...)` at least at the end of the run so this dict is present.
  - **Fallback:** **OptimizerResult** must include an optional **`method_summary`** (or **`extras`**) dict. The coordinate_then_random_direction optimizer returns **`OptimizerResult(..., method_summary={"mean_abs_grad_basis": ..., "mean_abs_grad_rand": ..., "n_basis_pairs": ..., "n_rand_pairs": ...})`**. optimize_driver, when persisting final_eval.json for this method, merges **result.method_summary** into the final_eval payload (using the final_* key names). If **ctx.get_method_state("coord_rd")** is None or incomplete, use **result.method_summary** so final_eval.json is still populated.

### 4.4 Plotting — read from standard eval record fields

- **Option chosen: (ii) separate plot file.** Create **`zero_order_grad_stats.png`** (only when method is `coordinate_then_random_direction`).
- **Data source:** Generate the plot by **scanning the history JSONL standard eval lines** (the same lines EvalContext writes: eval_id, x, status, f, etc.). **Extract** the fields **mean_abs_grad_basis**, **mean_abs_grad_rand**, **n_basis_pairs**, **n_rand_pairs**, **current_phase**, **current_iteration** (and **current_alpha** if needed) from each line. **Ignore** lines that do not have these fields. Do **not** require parsing special `"type": "coord_rd_grad_stats"` lines.
- **Content:** X-axis = evaluation index (eval_id) or a monotonic index over lines that have the grad fields. Two series: **mean_abs_grad_basis** and **mean_abs_grad_rand**. When n_basis_pairs == 0 (or field missing), omit or show nothing for basis; when n_rand_pairs == 0, omit or show nothing for rand. Legend or title: "Coordinate-then-random-direction: derivative magnitude estimates".
- **Optional:** Draw a vertical line at the eval index where **current_phase** changes from `"coordinate"` to `"random_direction"` (detect by scanning consecutive lines for a phase change).
- **Where implemented:** Add a function in `evolution/graph.py`, e.g. **`update_coord_rd_grad_stats_graph(history_path, output_dir)`**, which reads the JSONL file, filters to lines that contain the grad fields, and produces `zero_order_grad_stats.png`. The driver calls this when method is coordinate_then_random_direction (every N evals and at end, same as fitness graph).
- **Existing** `zero_order_fitness.png` is unchanged and still produced (fitness vs eval index).

---

## 5. Integration Points

### 5.1 Kconfig

- In `Kconfig`, add choice value `coordinate_then_random_direction` to the optimization method menu.
- Add a submenu or visible block that `depends on` this method, containing all CONFIG_COORD_RD_COORD_* and CONFIG_COORD_RD_RAND_* symbols listed in §2.2.

### 5.2 EvolutionConfig (kconfig_loader.py)

- Add fields to **EvolutionConfig** for **phase-specific** options when `optimization_method == "coordinate_then_random_direction"`. Populate from Kconfig:
  - **Phase 1 (coord):** coord_rd_coord_k, coord_rd_coord_alpha0, coord_rd_coord_alpha_min, coord_rd_coord_shrink, coord_rd_coord_num_coords_per_iter, coord_rd_coord_opportunistic, coord_rd_coord_improvement_eps; optional: coord_rd_coord_sample_with_replacement, coord_rd_coord_shuffle_each_iter, coord_rd_coord_max_coords_total.
  - **Phase 2 (rand):** coord_rd_rand_alpha0, coord_rd_rand_alpha_min, coord_rd_rand_shrink, coord_rd_rand_dir_dist, coord_rd_rand_dirs_per_iter, coord_rd_rand_improvement_eps; optional: coord_rd_rand_use_current_x_for_next_dir, coord_rd_rand_max_dir_pairs_total.

### 5.3 Dispatcher — explicit dispatch in optimize_driver

- **Requirement:** Dispatch **must** be explicit in **`evolution/optimize_driver.py`**. Do **not** rely on a central registry or `get_optimizer`; the plan does not assume they exist. Use either an **if/elif** chain or a **local dict** in optimize_driver that maps `cfg.optimization_method` (string) to the run callable.
- **Exact addition:** In **`evolution/optimize_driver.py`**, in the **Dispatch** section (where the chosen optimizer is invoked before the try block that runs it):
  - **Import:** `from .optimizers import coordinate_then_random_direction` (at top of file or inside the branch).
  - **Branch or dict entry:** Add a branch for `cfg.optimization_method == "coordinate_then_random_direction"` that sets `optimizer_fn = coordinate_then_random_direction.run` (or add this entry to a dispatch dict used for all methods).
  - **Call:** Invoke the optimizer the same way as others: `result = optimizer_fn(ctx, x0, bounds, cfg)`.
- In the **finally** block, when persisting state, if `cfg.optimization_method == "coordinate_then_random_direction"`, read **ctx.get_method_state("coord_rd")** and merge into zero_order_state.json (see §4.2); when writing final_eval.json, use ctx.get_method_state and/or result.method_summary (see §4.3).

### 5.4 Plot callback

- The driver already registers `on_eval_done` for fitness plot updates. For coordinate_then_random_direction, in the same **finally** block where it calls `update_zero_order_fitness_graph`, the driver also calls **`update_coord_rd_grad_stats_graph(history_path, output_dir)`** (or equivalent) when method is coordinate_then_random_direction. That function **reads** `zero_order_history.jsonl` and builds the grad-stats plot from **standard eval lines** that contain the mean_abs_grad_* (and related) fields (see §4.4). No separate grad_stats list from the optimizer is required.

---

## 6. Validation Rules (loader)

Validation runs in `_validate_zero_order_options` (or a dedicated validator for this method) when `optimization_method == "coordinate_then_random_direction"`. **Both** phase config sets are validated.

**Phase 1 (coordinate) symbols:**
- **coord_rd_coord_k:** K > 0.
- **coord_rd_coord_alpha0:** > 0.
- **coord_rd_coord_alpha_min:** > 0, and alpha_min <= alpha0.
- **coord_rd_coord_shrink:** in (0, 1).
- **coord_rd_coord_improvement_eps:** >= 0.
- **coord_rd_coord_num_coords_per_iter:** in [1, d] when d is known, or 0 meaning "all d".
- **coord_rd_coord_max_coords_total** (optional): if present, >= 0. Enforcement: both this cap and global max_evals are enforced (stop when either is hit).

**Phase 2 (random) symbols:**
- **coord_rd_rand_alpha0:** > 0.
- **coord_rd_rand_alpha_min:** > 0, and alpha_min <= alpha0.
- **coord_rd_rand_shrink:** in (0, 1).
- **coord_rd_rand_improvement_eps:** >= 0.
- **coord_rd_rand_dirs_per_iter:** >= 1.
- **coord_rd_rand_dir_dist:** in allowed set (e.g. `{"gaussian_unit", "rademacher_unit"}` or Kconfig choice).
- **coord_rd_rand_max_dir_pairs_total** (optional): if present, >= 0. Enforcement: both this cap and global max_evals are enforced.

---

## 7. Failure and Budget Interactions

- **BudgetExceeded and remaining budget before a ± pair:** Before calling **`ctx.evaluate_batch([x_plus, x_minus])`**, the optimizer **must** check **remaining_evals >= 2** (see §3.10). If not, stop cleanly and return OptimizerResult; the driver's **finally** block persists state (including last alpha, phase, iteration, derivative means/counts) so that resume or post-hoc analysis has a consistent snapshot.
- **BudgetExceeded (general):** When EvalContext raises BudgetExceeded (e.g. inside evaluate_batch), return OptimizerResult(best_x=ctx.best_x or x0, best_f=ctx.best_f or -inf, n_evals_used=ctx.n_evals, history_path=...). Persist state in the driver's finally with the latest coord_rd fields (see §4.2).
- **Per-eval failure:** EvalContext already applies CONFIG_EVAL_TIMEOUT_S, CONFIG_EVAL_MAX_RETRIES, CONFIG_EVAL_FAILURE_FITNESS. If a ± pair returns a failure fitness for one or both points, the optimizer should treat the pair as “no improvement” (do not update x from that pair; still update alpha shrink if the whole iteration had no improvement). When updating derivative means (mean_abs_grad_basis, mean_abs_grad_rand), use only pairs where both f_plus and f_minus are from successful evaluations; if either is a failure fitness, skip updating the mean for that pair (and do not increment the corresponding count).
- **Determinism and caching:** CONFIG_EVAL_DETERMINISTIC and cache behavior are unchanged; this method uses the same EvalContext and thus the same cache and failure policy.

---

## 8. Summary

| Item | Specification |
|------|----------------|
| Method name | `coordinate_then_random_direction` |
| Evaluation | Only via `ctx.evaluate_batch([x_plus, x_minus])` for every ± pair; no direct evaluate_x |
| Phase 1 | CONFIG_COORD_RD_COORD_K iterations; directions from **standard basis** (e_i); phase-specific alpha0/alpha_min/shrink/improvement_eps; optional MAX_COORDS_TOTAL cap |
| Phase 2 | Random **unit** directions (gaussian_unit or rademacher_unit); phase-specific alpha0/alpha_min/shrink/improvement_eps; alpha **reset to RAND_ALPHA0** at phase switch; optional MAX_DIR_PAIRS_TOTAL cap |
| Derivative stats | mean_abs_grad_basis (phase 1 only), mean_abs_grad_rand (phase 2 only); **ctx.set_extra_log_fields** (merged into every eval record); **ctx.set_method_state("coord_rd", ...)** for state/final_eval; **OptimizerResult.method_summary** as fallback for final_eval.json |
| State / final_eval | optimize_driver reads **ctx.get_method_state("coord_rd")** for zero_order_state.json and final_eval.json; state includes **coord_alpha_current**, **rand_alpha_current**, phase hyperparams (or Kconfig names); uses **result.method_summary** if ctx state missing |
| Checkpoint frequency | Optimizer updates **extra_log_fields** and **method_state** after each ± pair and at end of each iteration (see §4.2.1) |
| Plotting | New file `zero_order_grad_stats.png`; **data from standard eval record fields** in history JSONL (no coord_rd_grad_stats lines required) |
| Budget safety | Before each ± pair, check **remaining_evals >= 2**; stop cleanly and persist state if insufficient |
| Kconfig | New method choice + **phase-specific** CONFIG_COORD_RD_COORD_* and CONFIG_COORD_RD_RAND_* symbols; validation for both phase sets (see §6) |
| Integration | New module in evolution/optimizers; **explicit dispatch in optimize_driver** (if/elif or dict); state/history fields; graph reads history path |
