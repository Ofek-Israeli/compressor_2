# Genetic Evolution Plan: DEAP-Based Structured Evolution of Steering Deltas

This document describes the **only** evolution loop for compressor_2: a **DEAP**-based genetic evolution of steering deltas. There is no separate “reflective” or legacy evolution mode; **stick to the DEAP library** as much as possible. Fitness is computed on a fixed training set; mutation is reflector-based (driver pre-selects one target cluster, reflector proposes a new delta + scratchpad); crossover is weighted average of deltas by fitness and reflector merge of scratchpads. GPU is used for fitness evaluation; parallelism is applied where the hardware allows.

---

## 1. Overview

- **Framework:** DEAP (`deap`). Use `creator` / `toolbox` for individuals (delta vectors), fitness (single objective: weighted shortness + correctness), and operators (mutation, crossover, selection).
- **Individual:** One **delta vector** — logically a dict mapping cluster id (string `"0"`, `"1"`, …) to float, with fixed keys from initial deltas. **Canonical cluster id order:** `sorted(expected_cluster_ids)` from initial deltas keys (use numeric order, e.g. `sorted(keys, key=int)`), so dict↔list conversion is deterministic. **DEAP individual:** The individual is the delta list (`List[float]` in this fixed order) and **also carries a string attribute** `ind.scratchpad`. So the DEAP individual type is list-derived (the list holds the deltas); it has an attribute `scratchpad`. The population is a list of such individuals; **no parallel scratchpad structure** is used. We convert `ind` to/from dict for reflector and processor generation.
- **Scratchpad:** The scratchpad (not “CoT”) is a string **accumulated along an evolution path**, stored on the individual as `ind.scratchpad`. **Initial population:** set `ind.scratchpad = ""` for every initial individual. On **mutation**, the reflector receives the current `ind.scratchpad` and returns an updated string; the mutated individual’s `ind.scratchpad` is set to that value. On **crossover**, the reflector is given both parents’ `ind.scratchpad` and the parents’ fitness values and returns a merged scratchpad string for the child; set `child.scratchpad` to that value. Conflicting claims: reflector favors the better parent’s claims (evidence-supported) using fitness.
- **Training set:** The **same indices** as the existing pool (e.g. `FINANCEBENCH_POOL_INDICES`). No separate training set config. Fitness is always computed on this same training set.
- **Fitness function:**  
  `fitness = lambda_shortness * shortness_score + lambda_correctness * correctness_ratio`  
  where:
  - `shortness_score = 1 / (1 + mean_tok_len / shortness_scale)` (unchanged).
  - `correctness_ratio = num_correct / len(training_set)`.
  - `lambda_shortness` and `lambda_correctness` are **configurable in Kconfig** only; **no defaults** — if not set, raise an error (see §2).
- **Mutation:** Reflector-based. For a given individual (delta vector):
  1. We need **training-set outputs** generated with that delta vector, plus per-example **correctness** and **correctness explanation**. **Mutation of offspring:** Children are produced by crossover and have not been evaluated yet. Use **Option A:** Evaluate the child first, then mutate using the child’s own training outputs. After mutation we **re-evaluate** the mutated individual. So mutated offspring incur **two** fitness evaluations (evaluate → mutate → evaluate again).
  2. **Target cluster selection:** Target cluster selection is **round-robin** over the canonical cluster-id order (sorted keys, key=int). The **driver** owns the round-robin state. The DEAP-registered **`mutate(ind)`** function is created as a **closure** over a driver-owned mutable **`rr_state`** object that contains `cluster_ids` and the current `rr_idx` (0..M−1). On each call, `mutate(ind)` reads `c_target = rr_state.cluster_ids[rr_state.rr_idx]`, advances `rr_state.rr_idx = (rr_state.rr_idx + 1) % M`, and uses `c_target` in the reflector call. `rr_idx` is persisted in `evolution_state.json` so resume continues the same cycle. The reflector is told explicitly which cluster to mutate and that it must propose a new, different delta value for that cluster.
  3. Call the **reflector** with: cluster descriptions, **current delta vector**, **current scratchpad** (see below), the **target cluster id**, and training responses with correctness and explanations. Ask it to **propose a new delta value for the target cluster** to improve shortness without harming (or while improving) correctness.
  4. **OpenAI API:** Use JSON schema enforcement so the reflector returns **delta** and **scratchpad** (string). The reflector updates the scratchpad with what it learnt this step; the mutated individual carries this **accumulated scratchpad** along its evolution path.
  5. **Mutation result:** `mutate(ind)` operates **in-place** on the (already cloned) offspring passed to it: replace the target cluster’s delta in `ind` with the reflector’s value and set `ind.scratchpad` to the reflector’s returned scratchpad. All other clusters remain unchanged. No separate “copy” of the individual is made for mutation itself; any copy is only for logging or the evolution tree (see below).
- **Crossover:** **Weighted average** of two delta vectors by fitness. Our driver calls `mate(ind1, ind2)` once per parent pair and expects **one child** per call (§4). For two individuals `a` and `b` with fitness values `f_a` and `f_b`, produce child deltas:  
  `child[c] = (f_a * a[c] + f_b * b[c]) / (f_a + f_b)`  
  for each cluster `c`. If both fitnesses are zero, use plain average. The **child’s scratchpad** is produced by a **reflector merge call**: provide both parents’ scratchpads **and parents’ fitness values**; ask the reflector to merge them (evidence-supported wins on conflicts; reflector can favor the better parent’s claims using fitness).
- **DEAP operators (scratchpad access):** **evaluate(ind)** receives the individual; it reads `ind` for the delta list (convert to dict for processor/run) and can read `ind.scratchpad` for logging or context if needed; it does not write `ind.scratchpad`. **mutate(ind)** is registered as a **closure** over a driver-owned mutable **`rr_state`** object (containing `cluster_ids` and `rr_idx`). On each call, `mutate(ind)` reads `c_target = rr_state.cluster_ids[rr_state.rr_idx]`, advances `rr_state.rr_idx = (rr_state.rr_idx + 1) % M`, and uses `c_target` in the reflector call; it then updates the target cluster's delta coordinate in `ind` and sets `ind.scratchpad` to the reflector's returned scratchpad. **mate(ind1, ind2)** computes the fitness-weighted-average delta list for the child and sets `child.scratchpad` from a reflector merge call using both parents' scratchpads (`ind1.scratchpad`, `ind2.scratchpad`) and their fitness values.
- **GPU and parallelism:** Fitness evaluation uses the GPU (generate processor → SGLang → training set → correctness → fitness). See **§5** for single-GPU (sequential) and optional multi-GPU.
- **Evolution loop (high level):**
  - **How to run:** The DEAP GA evolution is triggered via the single CLI entry point: `PYTHONPATH=/workspace python3 -m compressor_2 evolve --config .config`. The command **`compressor_2 evolve`** always runs the DEAP-based evolution loop described in this plan. There are **no mode flags or legacy evolution paths**; this command is the **only supported evolution command**. The `--config .config` argument points to the Kconfig-generated config file used by the loader.
  1. **Load config:** Kconfig (including training set indices, lambda_shortness, lambda_correctness, shortness_scale, reflector, paths, SGLang settings). Non-GA parameters (shortness_scale, cluster descriptions, reflector, SGLang, correctness) come from existing Kconfig; §2 lists only new GA-specific symbols.
  2. **Training set:** Resolve training example indices and keep them fixed for the whole run.
  3. **Initial population:** **One individual cloned** — take the single initial deltas from config and clone it **population_size** times (no perturbation). Set `ind.scratchpad = ""` for every initial individual. When creating or cloning individuals, ensure the `scratchpad` attribute is set (and that cloning copies it). Population size from Kconfig.
  4. **Generation loop (exact flow, configurable via Kconfig):**
     - **Offspring count:** `population_size - elitism_size` so that next population = elites + offspring = **population_size**. The flow (selection method, number of parents, crossover→mutation→evaluate order) is **configurable via Kconfig** (see §2).
     - **Canonical flow:** (1) **Evaluate** all individuals in the population; cache training-set results per individual. **Elitism timing:** select the top CONFIG_ELITISM_SIZE elites from the fully evaluated current population immediately after step (1) “Evaluate all individuals” and BEFORE creating or mutating any offspring; then clone those elites (`toolbox.clone(elite)`) into the next population. (2) **Selection:** select exactly **2 × (population_size − elitism_size)** parents (to be paired for offspring). Offspring must be created from cloned parents before any in-place operations (§8). When **CONFIG_SELECTION = "truncation"**: (a) sort the population by fitness in descending order; (b) keep only the top **CONFIG_TRUNCATION_TOP_K** individuals as the parent pool; (c) sample parents from this pool **with replacement** until 2×(population_size − elitism_size) parents are selected. When CONFIG_SELECTION = "tournament": use CONFIG_TOURNAMENT_SIZE for tournament selection to select the same number of parents. (3) **Crossover:** pair parents, do `(population_size - elitism_size)` crossovers → that many children. With probability CONFIG_CXPB apply weighted-average crossover and reflector merge for scratchpad. **If crossover is not applied** (probability 1−CONFIG_CXPB), the child is a **clone of exactly one parent chosen at random (50/50 between Parent A and Parent B)**. The clone copies both the delta list and the scratchpad verbatim. This is the default DEAP-style fallback and ensures exactly (population_size − elitism_size) offspring each generation. **Offspring list then mutation→evaluation:** After we create the full offspring list of exactly (CONFIG_POPULATION_SIZE − CONFIG_ELITISM_SIZE) children (via mate when crossover applies, otherwise 50/50 parent clone), we iterate over that offspring list. For each child: run one full fitness evaluation (the child is new and has no fitness yet); then, with probability CONFIG_MUTPB, apply reflector mutation using the child’s own training outputs and immediately re-run a second full fitness evaluation; otherwise keep the single evaluation. Fitness must be invalidated whenever an offspring is created or its genome is modified (§8). (4) **Mutation** (per-child, as above). (5) **Replace:** next population = elitism (best CONFIG_ELITISM_SIZE individuals) + these offspring. Elites are **copied** (cloned) into the next population, not carried by reference (§8); cloned elites keep their valid fitness (no invalidation needed).
     - **Logging / persistence:** Maintain a **best-ever** individual (global best across all evaluated individuals so far). Update `best_individual` and `best_fitness` whenever a newly evaluated individual’s fitness exceeds the stored best. Save best-ever (deltas + scratchpad) and best processor; persist resume state (**§2**). Log prompts sent to the reflector for each mutation and merge (**§8**). Graphs: fitness vs generation; existing evolution images + **evolution tree** (parent–child). **Ctrl+C:** persist and exit; next run resumes (§2).
  5. **SGLang lifecycle:** Same as today: before each fitness evaluation that needs the processor, we need the processor file; so generate-processor from the individual’s delta vector, then (if not already running) start SGLang with that processor, run training set, run correctness, compute fitness. For multiple individuals we either reuse one SGLang run per individual (regenerate processor, restart SGLang, run training set) or batch where possible. So **one SGLang server, one processor at a time**: for each individual we regenerate processor, (re)start SGLang, run training set, then we can stop SGLang before the next individual to free GPU for the next generate-processor if needed.

---

## 2. Kconfig Additions

- **Scope of §2:** All non-GA parameters referenced by the evolution loop (e.g. CONFIG_SHORTNESS_SCALE, CONFIG_CLUSTER_DESCRIPTIONS_PATH, reflector model/API config, SGLang/server config, and correctness/evaluation config) are taken from **existing Kconfig options** already in the repo; §2 lists **only the new GA-specific symbols**.
- **No defaults for genetic evolution options:** All options below are required. Use **sentinel defaults** (e.g. `-1` for ints, `""` for strings); **loader raises at startup** if any genetic evolution option is still sentinel (no silent defaults).
- **Fitness weights:** `CONFIG_LAMBDA_SHORTNESS`, `CONFIG_LAMBDA_CORRECTNESS` (float). Validation: both in [0, 1]; can normalize to sum to 1.
- **Training set:** Same as pool indices (§1). Use existing `CONFIG_FINANCEBENCH_POOL_INDICES`; if pool is **"all"**, full dataset = training set.
- **Initial deltas:** `CONFIG_INITIAL_DELTAS_PATH` (JSON: `{ "<cluster_id>": <float>, ... }`). Used for initial population (one individual cloned) and canonical cluster order (§1).
- **DEAP / GA:** `CONFIG_POPULATION_SIZE`, `CONFIG_NGEN`, `CONFIG_CXPB`, `CONFIG_MUTPB`, `CONFIG_ELITISM_SIZE` (int/float; all required via sentinel). **CONFIG_NGEN:** Generations are 0-based. CONFIG_NGEN is the **number of generations to run**, so the loop runs for generation indices **0 .. CONFIG_NGEN−1** inclusive. **Selection:** `CONFIG_SELECTION` (string: `"tournament"` or `"truncation"`). If tournament: `CONFIG_TOURNAMENT_SIZE` (int, required). If truncation: **CONFIG_TRUNCATION_TOP_K** (int, required) — number of top individuals to form the parent pool. **Truncation selection (when CONFIG_SELECTION = "truncation"):** (1) Sort the population by fitness in descending order. (2) Keep only the top K individuals (K = CONFIG_TRUNCATION_TOP_K) as the parent pool. (3) Sample parents from this pool **with replacement** until **2 × (CONFIG_POPULATION_SIZE − CONFIG_ELITISM_SIZE)** parents are selected, then pair them for offspring generation. This matches the typical DEAP composition: use `selBest(pop, K)` to form the pool, then `selRandom(pool, N)` (with replacement) to draw N = 2×(population_size − elitism_size) parents. **Selection parameter validation:** Validate **CONFIG_TRUNCATION_TOP_K ≥ 2** (otherwise truncation cannot form parent pairs). Optionally, for stricter checks, validate **CONFIG_TRUNCATION_TOP_K ≤ CONFIG_POPULATION_SIZE**. Note: **top_k ≥ 2×(population_size − elitism_size) is NOT required** because parents are sampled with replacement; the only hard requirement is **top_k ≥ 2**. For tournament selection: validate **CONFIG_TOURNAMENT_SIZE ≥ 2** and **CONFIG_TOURNAMENT_SIZE ≤ CONFIG_POPULATION_SIZE**. Flow (offspring count = population_size − elitism_size, etc.) configurable via these symbols.
- **Resume path and format:**
  - **Path:** Single state file **`evolution_state.json`** under the configured output directory (e.g. `output_dir/evolution_state.json`).
  - **Exact contents:** (a) `generation` (int), (b) `population` (list of `{deltas: {cluster_id: float}, scratchpad: string}`), (c) `best_individual`: **best-ever individual** encountered so far (global best across all evaluated individuals up to the saved generation), stored as `{deltas, scratchpad}`. (d) `best_fitness`: the corresponding best-ever fitness (optional but recommended). (e) **`rr_idx`** (int): round-robin index for mutation target cluster selection (0..M−1); persisted so resume continues the same cycle. On resume, the driver restores **`rr_state.rr_idx`** from the saved `rr_idx` before registering or using **`mutate(ind)`**. Do **not** persist cached training results; on resume, **re-evaluate the whole population** at the start of the next run, then **recompute/update the global best** (best_individual and best_fitness) from the re-evaluated population and any previously stored best.

---

## 3. Reflector Mutation: API and Schema

- **Inputs to reflector (for mutation):** As in §1: cluster descriptions, current delta vector, current scratchpad, the **target cluster id** (pre-selected by the driver), and training-set responses (example_id, llm_answer, correctness, explanation). Instruction: propose a **new delta value** for the target cluster; all other clusters remain unchanged.
- **Output schema (OpenAI JSON schema):**
  - `delta` (number): the new value for the target cluster.
  - `scratchpad` (string): the **accumulated scratchpad** for this evolution path (reflector appends/updates what it learnt this round; this is passed back to the reflector in the next mutation for this lineage).
- **Mutation implementation:** **mutate(ind)** operates **in-place**: it receives the DEAP individual (delta list + `ind.scratchpad`). **Round-robin plumbing:** the DEAP-registered `mutate(ind)` is created as a **closure** over a driver-owned mutable **`rr_state`** object that contains `cluster_ids` and the current `rr_idx`. On each call, `mutate(ind)` reads `c_target = rr_state.cluster_ids[rr_state.rr_idx]`, advances `rr_state.rr_idx = (rr_state.rr_idx + 1) % M`, and uses `c_target` in the reflector call. The reflector contract is unchanged: the target cluster is provided; the reflector must change its delta. Parse reflector response; validate `delta` (numeric). Update `ind[target_cluster_index]` to the reflector’s returned delta. Set `ind.scratchpad` to the reflector’s returned scratchpad. (Convert `ind` to/from dict for reflector and processor as needed.)

### 3.1 Reflector mutation: full prompt templates

**System message:**

```
You are a reflector for a steering system that adjusts LLM outputs by cluster. Your task is to propose a new delta value for a specific target cluster (chosen by the driver) to reduce verbosity without harming correctness. You receive: (1) cluster descriptions, (2) the current delta vector and scratchpad for this individual, (3) the target cluster to mutate, (4) training-set responses with per-example correctness and explanation. The target cluster is fixed for this call; you MUST change its delta to a new value (different from the current value). All other clusters remain unchanged. Output only a JSON object with two fields: "delta" (number), "scratchpad" (string — your updated scratchpad for this evolution path, appending what you learnt this step). Negative deltas discourage tokens in that cluster. No other commentary.
```

**User message (template):**

```
## Cluster descriptions
{cluster_descriptions_json}

## Current delta vector
{current_deltas_json}

## Current scratchpad
{current_scratchpad}

## Target cluster to mutate (must change)
{target_cluster_id}

## Training-set responses (correctness and explanation per example)
{all_responses_with_correctness}

---
Propose a new delta value for the target cluster above to improve shortness without harming correctness. The delta MUST be different from the current value for that cluster. All other clusters remain unchanged. Update the scratchpad with what you learnt. Output only: {"delta": <float>, "scratchpad": "<string>"}.
```

**OpenAI response schema (strict JSON):** `{"type": "object", "properties": {"delta": {"type": "number"}, "scratchpad": {"type": "string"}}, "required": ["delta", "scratchpad"], "additionalProperties": false}`.

---

## 4. Crossover Implementation

- **Custom mate operator:** Implement and register a **custom** `mate(ind1, ind2)` in the DEAP toolbox. Our custom evolution driver expects **`mate(ind1, ind2)` to return a single child individual (not two)**. We generate exactly (CONFIG_POPULATION_SIZE − CONFIG_ELITISM_SIZE) children by pairing the 2×(pop−elitism) selected parents into that many pairs and calling `mate` once per pair. This is a custom driver (we do not use DEAP’s stock algorithms that assume mate returns two individuals). Do **not** use DEAP’s standard `tools.cx*` operators (e.g. `cxTwoPoint`, `cxBlend`) because they do not incorporate fitness or scratchpad merge. The custom `mate(ind1, ind2)` must: read `ind1.fitness.values[0]` and `ind2.fitness.values[0]`; compute the fitness-weighted-average delta vector for the child; call the reflector to merge `ind1.scratchpad` and `ind2.scratchpad` using the parents’ fitness values; set `child.scratchpad` to the merged result; and **return the resulting child individual**. After constructing the child (deltas + merged scratchpad), `mate(ind1, ind2)` must invalidate the child's fitness (`del child.fitness.values`) before returning the child.
- **Delta vector:** As in §1: weighted average by fitness; if `f_a + f_b == 0`, use `(a[c] + b[c]) / 2`. DEAP: store as list of floats (same order as cluster ids); `child_list[i] = (f_a * a[i] + f_b * b[i]) / (f_a + f_b)` (fitness from `.fitness`). **When crossover is not applied (probability 1−CONFIG_CXPB):** the child is a clone of exactly one parent chosen at random (50/50 between Parent A and Parent B); copy both the delta list and the scratchpad verbatim (`child.scratchpad` = that parent’s scratchpad). See §8.
- **Scratchpad merge:** **mate(ind1, ind2)** produces a child individual (delta list). The child's scratchpad is not averaged: call the reflector with `ind1.scratchpad`, `ind2.scratchpad`, and both parents' fitness values; use the templates below. Set `child.scratchpad` to the reflector's returned string. Schema: reflector returns only `{"scratchpad": "<string>"}`.

### 4.1 Reflector merge: full prompt templates

**System message:**

```
You are a reflector that merges two scratchpads from two parent individuals in an evolutionary run. You receive two scratchpads and the fitness value of each parent. Your task is to produce a single merged scratchpad for their child. The merged scratchpad should combine the insights from both parents. If there are conflicting claims between the two scratchpads, prefer the claim from the parent with the higher fitness (more evidence-supported). Output only a JSON object with one field: "scratchpad" (string). No other commentary.
```

**User message (template):**

```
## Parent A (fitness: {fitness_a})
{scratchpad_a}

## Parent B (fitness: {fitness_b})
{scratchpad_b}

---
Merge these two scratchpads into one for the child. When claims conflict, favor the higher-fitness parent. Output only: {"scratchpad": "<merged string>"}.
```

**OpenAI response schema (strict JSON):** `{"type": "object", "properties": {"scratchpad": {"type": "string"}}, "required": ["scratchpad"], "additionalProperties": false}`.

---

## 5. GPU and Parallelism (Concrete)

- **Assume one GPU** by default. **Seize the GPU** and **parallelize as much as possible** within that constraint (e.g. sequential fitness evaluations that each use the GPU fully; no multi-GPU required).
- **Single GPU:**  
  - For each fitness evaluation: (1) generate-processor from individual’s deltas, (2) stop SGLang if running, (3) start SGLang with new processor, (4) run training set via financebench_runner, (5) run correctness, (6) compute fitness. Evaluations are **sequential** (one at a time).
  - Use DEAP with a sequential `map` so that one evaluation runs at a time and fully uses the GPU.
- **Multiple GPUs (optional):**  
  - If extended later: workers in a pool, each assigned a GPU index; CONFIG_N_GPUS or similar; parallel map when N_GPUS > 1.

---

## 6. Implementation Checklist (for implementation phase)

1. **Dependencies:** Add `deap` to `requirements.txt`.
2. **Kconfig:** §2. All genetic evolution options required (sentinel → loader raises at startup). Selection: CONFIG_SELECTION (string), CONFIG_TOURNAMENT_SIZE when tournament, truncation top-k when truncation.
3. **Kconfig loader:** Single evolution path (no mode flags). Validate genetic options; raise if any sentinel.
4. **Fitness:** `evaluate_individual(...)`: generate processor, (re)start SGLang, run training set with correctness, return (shortness_score, correctness_ratio); cache training results for reflector. Fitness = lambda_shortness * shortness + lambda_correctness * correctness. DEAP single-objective maximization.
5. **Individual representation:** DEAP individual = delta list (`List[float]` in canonical order) with attribute `ind.scratchpad` (str). Define the individual type via `deap.creator`: list-derived with fitness (e.g. `creator.create("Individual", list, fitness=creator.FitnessMax)`), which permits arbitrary attributes. On initialization and when creating the initial population, set `ind.scratchpad = ""`; ensure cloning (e.g. when copying one parent or creating offspring) copies the `scratchpad` attribute. Set `toolbox.clone` to `copy.deepcopy` (or equivalent) per §8 so that delta list, `scratchpad`, and `fitness` are deep-copied. Population = list of such individuals; no parallel scratchpad structure. deltas_dict_to_list / deltas_list_to_dict for reflector and processor. Initial population §1 (clone one, empty scratchpads). Canonical order §1.
6. **Mutation:** §3. Evaluate child → reflector (§3.1) → re-evaluate if mutated (two evals for mutated offspring).
7. **Crossover:** §4. Implement and register a **custom** `mate(ind1, ind2)` in the toolbox (do not use DEAP’s `tools.cx*`); it must use parents’ fitness and reflector merge for scratchpad. Weighted average + reflector merge (§4.1). Register with DEAP.
8. **Evolution driver:** Single entry point: `python3 -m compressor_2 evolve --config .config` (see §1 “How to run”). `compressor_2 evolve` always runs the DEAP loop; no mode flags or legacy paths. Load config from the file given by `--config` (Kconfig-generated, used by the loader), build training set from pool (§1), initial population §1, toolbox from Kconfig, DEAP with elitism. Track **best-ever** individual (global best); update whenever a newly evaluated individual exceeds the stored best. **Resume:** §2 (persist evolution_state.json; on resume re-evaluate population, then recompute/update global best from re-evaluated population and stored best). **Ctrl+C:** persist and exit.
9. **Logging and graphs:** Best/mean fitness per generation; fitness vs generation plot; **best-ever** deltas and processor (update best-ever whenever a new individual exceeds it). **Reflector prompts:** §8. Evolution tree image (parent–child across generations).

---

## 7. Summary of Decisions

- **Scratchpad** (§1): stored on the individual as `ind.scratchpad`; no parallel structure. Accumulated along path; empty initially; reflector updates on mutation, merges on crossover (evidence-supported).
- **Training set** (§1): same as pool indices; "all" = full dataset.
- **Kconfig** (§2): all genetic options required (sentinel → raise); selection string + tournament size or truncation top-k.
- **Resume** (§2): `evolution_state.json`; generation, population, **best_individual** (best-ever / global best), best_fitness (optional but recommended), **rr_idx** (round-robin mutation target index); re-evaluate on resume then recompute global best; Ctrl+C persist and exit.
- **Canonical order** (§1): `sorted(expected_cluster_ids, key=int)`.
- **Reflector prompts** (§8): log exact system+user messages per call.
- **GPU** (§5): one GPU default; sequential evals.

---

## 8. Implementation Notes (for implementer)

- **Initial deltas / canonical order:** §2. Use `sorted(initial_deltas.keys(), key=int)`.
- **Fitness invalidation rule (DEAP):** Any time we create a new individual or modify its genes, we must delete its fitness so it will be recomputed. Concretely: (1) After creating a child (either via mate/crossover or via 50/50 parent cloning): `del child.fitness.values`. (2) After mutating an individual in-place: `del ind.fitness.values`. This prevents stale fitness from being reused (e.g., cloned parents retaining parent fitness, mutated genomes retaining old fitness).
- **In-place operator safety (DEAP):** DEAP operators may modify individuals in-place. Therefore, offspring must always be created as clones of the selected parents before applying any in-place operations (crossover/mutation). Do not run mate/mutate directly on parent objects that still live in the current population; always operate on cloned copies so parents remain unchanged for elitism, logging, and correctness. Use `toolbox.clone(parent)` (or equivalent) prior to mate/mutate; only the cloned offspring are modified.
- **Clone semantics:** Set `toolbox.clone` to `copy.deepcopy` (or an equivalent deep-clone function) so that cloning copies the delta list contents, the `ind.scratchpad` attribute, and the `ind.fitness` object. This ensures parent cloning, offspring creation, and elitism cloning behave correctly. We still invalidate fitness for newly created offspring per the fitness invalidation rule; elites keep their valid fitness because the genome is unchanged.
- **Elitism copying:** When forming the next population, take the top CONFIG_ELITISM_SIZE elites from the evaluated current population and insert `toolbox.clone(elite)` copies into the next population. Do not carry elite objects by reference; cloning prevents accidental in-place modification of elites by later operations. Cloned elites keep their valid fitness (no invalidation needed since the genome is unchanged).
- **CXPB (no-crossover fallback):** With probability **1−CONFIG_CXPB**, crossover is not applied: the child is a **clone of exactly one parent chosen at random (50/50 between Parent A and Parent B)**. The clone copies both the delta list and the scratchpad verbatim. With probability CONFIG_CXPB, apply weighted-average crossover + reflector merge. This is the default DEAP-style GA fallback and ensures exactly (population_size − elitism_size) offspring each generation. Parent choice for the clone is **random 50/50** unless overridden later.
- **Reflector validation:** Mutation: treat the reflector's `delta` as **invalid** when (1) it is non-numeric, or (2) it is equal to the current delta value for the selected target cluster `c_target` (use exact equality or an `eps` tolerance for floats). On invalid output: retry the reflector call once; if the second attempt is still invalid (non-numeric or unchanged), skip mutation for that child. This ensures the reflector cannot return the same delta and silently "not mutate". Merge: if invalid JSON or missing `scratchpad`, retry once; else e.g. first parent's scratchpad.
- **Resume:** Generations are 0-based; CONFIG_NGEN is the number of generations to run (loop 0..CONFIG_NGEN−1 inclusive). Persist at end of each generation and on SIGINT (§2). **Round-robin state:** Include **`rr_idx`** in the persisted state (`evolution_state.json`). On resume, the driver **restores `rr_state.rr_idx`** from the saved `rr_idx` **before** registering or using **`mutate(ind)`**, so the mutation target cluster cycle continues from where it left off.
- **Evolution tree:** Track parent refs when creating offspring; build graph (generation, index) → child; render and save (e.g. `evolution_tree.png` in output_dir) (§1/§6). Use the following **stable node-ID naming convention** so the graph is unambiguous, especially with the pre-mutation intermediate node. **Node IDs:** (1) For any individual in the **current population** at the start of generation g (right after “Evaluate all individuals”): `node_id = g{g}_pop{i}` where `i` is the 0-based index in the population list at that moment. (2) For each offspring created in generation g, for offspring index j in `0 .. (CONFIG_POPULATION_SIZE − CONFIG_ELITISM_SIZE − 1)`: pre-mutation node_id = `g{g}_child{j}_pre`. (3) If that offspring is mutated (MUTPB triggers), create a separate post-mutation node: post-mutation node_id = `g{g}_child{j}_mut`. (4) If that offspring is NOT mutated, treat the pre-mutation node as the final offspring node (no extra “final” name; final = `..._pre`). **Edges to record:** Parent links: `g{g}_pop{iA} -> g{g}_child{j}_pre` and `g{g}_pop{iB} -> g{g}_child{j}_pre` (or a single parent edge if no-crossover clone). Mutation link (only if mutated): `g{g}_child{j}_pre -> g{g}_child{j}_mut`. The individual that enters the next population is `g{g}_child{j}_mut` if mutated, else `g{g}_child{j}_pre`. This ID scheme is deterministic and derived only from (generation g, list index i/j, and stage suffix), so it remains stable across logging and resume. **Mutation and the tree (in-place consistency):** When MUTPB triggers for an offspring, first **snapshot** the pre-mutation state (deltas + scratchpad) for the `..._pre` node—either by recording a copy of the list and scratchpad, or by cloning once to represent the pre-mutation node. Then apply `mutate(ind)` **in-place** on the offspring to produce the post-mutation genome/scratchpad for the `..._mut` node. The post-mutation individual (after re-evaluation) is what enters the next population. Thus `mutate(ind)` always modifies the provided offspring in-place; “copy” is only for logging/tree snapshotting.
- **Reflector prompt logging:** Log exact system + user messages sent to the API (after substitution) for every mutation and merge call. Store under output_dir (e.g. `reflector_prompts/gen_N_mutation_<id>.txt`, `gen_N_merge_<id>.txt`, or single file with separators and metadata: generation, call type, indices).
