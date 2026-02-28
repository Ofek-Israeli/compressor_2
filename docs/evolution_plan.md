# Evolution Plan: Steering Deltas for Cluster-Based Logit Processor

This document describes the plan for a new module under `compressor_2` that **evolves** the per-cluster steering magnitudes (deltas) in a loop: run FinanceBench minibatches with the current processor, send **all** minibatch responses **with per-example correctness** to a reflector (GPT-4o), get updated deltas, regenerate the processor, and repeat. The goal is to reduce padding and verbosity **while maintaining or improving correctness** (composite score: 0.4×shortness + 0.6×correctness ratio).

---

## 1. Overview

- **Inputs (fixed):**
  - `outputs/labels_kmeans.joblib`, `outputs/embeddings.npy` (or PCA joblib), `outputs/cluster_descriptions.json`
  - Initial deltas: e.g. `outputs/deltas_examples.json`
  - FinanceBench JSONL path and optional pool of example indices (configurable)
- **Loop (each iteration):**
  1. **Stop SGLang server** (so the next step can use the GPU without OOM).
  2. **Generate processor:** from current deltas → run `generate-processor` → produce `processor.py`. This step **must run on GPU** (do not set `CUDA_VISIBLE_DEVICES`). If the GPU is unavailable or CUDA OOM occurs, **raise an error and terminate** the evolution (no CPU fallback).
  3. **Start SGLang server** (subprocess) with same args as user’s (e.g. `--model-path`, `--port 8000`, `--enable-custom-logit-processor`). Wait until healthy.
  4. **Run minibatch:** run FinanceBench on a **minibatch** of examples using `processor.py` (e.g. via financebench_runner). Collect `(example_id, llm_answer, question, ground_truth_answer)` per result.
  5. **Correctness:** for each minibatch result, run the **correctness evaluator** (learning_grammar/correctness.py) to get `is_correct` and `reasoning`. Attach these to each result.
  6. **Reflector call:** send to GPT-4o:
     - **All** minibatch responses with **per-example correctness and explanation** (not just k longest).
     - The **deltas that were used for this minibatch** (the current deltas that produced these responses), so the reflector can propose changes from a known baseline.
     - The contents of **cluster_descriptions.json** (so the model knows what each cluster represents).
     - An **evolving summary (CoT):** what the reflector has **learnt so far** across previous iterations (e.g. which clusters tended to add fluff, what delta changes helped, what to try next). The reflector **updates** this each iteration; the driver **serves** the accumulated evolving summary back to the reflector in later iterations.
     - **Instruction:** reduce verbosity **without harming correctness**; given the deltas used for this minibatch, propose updated deltas per cluster and update the evolving summary; output only a JSON object with `"deltas"` and `"summary"`.
     - Use **OpenAI JSON schema enforcement** (`response_format` with a JSON schema) so the API returns only valid JSON with required keys `"deltas"` and `"summary"` (see §4.3 and §4.4).
  7. **Update deltas:** parse reflector output → new `deltas` dict; validate (see §4.4). Write to `deltas_current.json`.
  8. **Track best:** compute **composite score** = 0.4×shortness_score + 0.6×correctness_ratio (shortness_score = 1/(1 + mean_tok_len/scale), scale from Kconfig). If this is **higher** than the best so far, copy current deltas to `deltas_best.json`.
  9. **Logging and graph:** record per iteration: mean_token_length, correctness_ratio, shortness_score, composite_score. Log the full reflector message per iteration. Update **matplotlib** plot: composite score vs iteration (and optionally correctness ratio / mean length); save as `evolution_lengths.png`.
  10. **Next iteration** (or exit if max iterations reached).
- **On Ctrl+C (SIGINT):** catch signal, ensure `deltas_best.json` (and optionally `processor_best.py`) are written, then exit. User can then use the best processor from the evolution output directory.

---

## 2. SGLang Server and GPU Memory

- **Constraint:** When the SGLang server is running on the GPU, `generate-processor` (sentence-transformers embedding of full vocab) causes CUDA OOM. When `generate-processor` uses the GPU, the server cannot run on the same device.
- **Approach:** The evolution driver **controls the SGLang server process**:
  - **Before** calling `generate-processor`: **stop** the SGLang server (kill the subprocess if the driver started it, or document that the user must stop an externally started server).
  - **After** `generate-processor` finishes: **start** the SGLang server as a **subprocess** (with configurable command line, e.g. `python -m sglang.launch_server --model-path ... --port 8000 ... --enable-custom-logit-processor`), then wait until healthy: e.g. **GET the health endpoint** (SGLang’s actual health URL, e.g. `http://localhost:8000/health` or as in SGLang docs) with a **timeout and retry loop** (e.g. retry every few seconds until success or max attempts).
  - **Before** running the FinanceBench minibatch: ensure the SGLang server is **running** (the driver started it in step 3).

---

## 2.5 Special handling of clusters 0 and 1 when generating the custom logit processor

When turning (evolved) deltas into a custom logit processor (the `generate-processor` step), **clusters 0 and 1 must be handled differently** from the rest. They correspond to the fixed special clusters (see `add_special_clusters.py` and `cluster_descriptions.json`):

- **Cluster 0 (EOS/EOT):** Map **only** the model’s EOS/EOT token IDs to `deltas["0"]`. Do **not** assign these tokens via the embed → PCA → k-means path. The generator must detect EOS/EOT token IDs (e.g. from the tokenizer) and apply `deltas["0"]` to them exclusively.
- **Cluster 1 (Numbers, arithmetic symbols):** Map an **explicit set** of token IDs (digits, `+`, `-`, `*`, `/`, `$`, `%`, `=`, etc.) to `deltas["1"]`. Do **not** assign these tokens via the embed → PCA → k-means path. The generator must obtain this set (e.g. by decoding each vocab token and classifying it as digit/symbol, or from a fixed list of token IDs for the given tokenizer) and apply `deltas["1"]` only to those IDs.

All other clusters (keys `"2"` … `"11"` or as in `cluster_descriptions.json`) are filled from the normal path: decode vocab → embed → PCA → k-means → `deltas[str(cluster + 2)]`. So when generating the processor, the implementation must branch: (1) EOS/EOT → `"0"`; (2) number/symbol tokens → `"1"`; (3) everything else → k-means + offset.

---

## 3. Configuration (Kconfig)

- Add a **Kconfig** (and a `.config` / defconfig) under `compressor_2` so that:
  - **FinanceBench pool:** path to FinanceBench JSONL; and either a list of **example indices** (0-based) to form the pool, or “all” (use all examples). Each iteration the minibatch is **randomly sampled** from this pool (reproducibility not required).
  - **Minibatch size:** number of FinanceBench examples per iteration.
  - **SHORTNESS_SCALE:** scale (tokens) for shortness_score = 1/(1 + mean_tok_len/scale); configurable in Kconfig.
  - **Correctness:** model and tolerance for per-example correctness (learning_grammar); Minions repo path.
  - **Number of evolution iterations:** max iterations; can also stop early if desired later.
  - **Reflector:** model name (default gpt-4o), API key from env (e.g. `OPENAI_API_KEY`).
  - **Paths:** output directory for evolution (e.g. `outputs/evolution/`), paths to `labels_kmeans.joblib`, `cluster_descriptions.json`, initial deltas, embeddings/PCA, and to the generated `processor.py` / `processor_best.py`.
  - **SGLang server command:** full command line (or key args) to start the SGLang server (port, model path, `--enable-custom-logit-processor`, etc.).
  - **financebench_runner config:** path to the YAML config used when calling the runner (or equivalent in-process client) for the minibatch.
- Load Kconfig in the evolution module (e.g. via a small `kconfig_loader.py` or reuse a pattern from `learning_grammar`).
- Provide **`make menuconfig`**: a **Makefile** (e.g. under `compressor_2/`) must exist such that running `make menuconfig` launches the Kconfig GUI (e.g. ncurses-based menu), which writes `.config`.

---

## 4. Reflector Prompt and JSON Schema

### 4.1 Inputs to the reflector

- The **k longest** LLM responses from the minibatch (optionally truncated to a max character limit to avoid token limits).
- The **deltas that were used for this minibatch** (current iteration's deltas, as a JSON object), so the reflector knows the baseline it is adjusting from.
- The full **cluster_descriptions.json** so the model knows what each cluster represents.
- An **evolving summary (CoT):** what the reflector has **learnt so far** across previous iterations. The reflector must **update** this evolving summary each round; the driver appends the reflector's update to the accumulated text and **serves** it back to the reflector in later iterations. The reflector returns its update in a `"summary"` field in its JSON (see §4.3). First iteration: CoT can be empty or "First iteration; no prior learnings."
- **No example deltas:** do not pass `deltas_examples.json` or any example deltas so the reflector does not copy a template.

### 4.2 Full prompt template

**System message:**

```
You are a reflector for a steering system that adjusts LLM outputs by cluster. Your task is to spot padding, verbosity, and fluff in the given model responses and propose per-cluster steering deltas (a float per cluster id) so that future responses become razor-sharp. You will receive: (1) cluster descriptions (what each cluster id means), (2) the deltas that were used for this minibatch (the baseline you are adjusting from), (3) all minibatch model responses with per-example correctness and explanation, and (4) the evolving summary of what you have learnt so far (CoT), which you have updated in previous iterations. You must output a JSON object with two fields: "deltas" (object mapping each cluster id string "0", "1", ... to a float) and "summary" (your update to the evolving summary: 2–4 sentences on what you learnt this round—the driver will append this and serve the accumulated evolving summary back to you in later iterations). Negative deltas discourage tokens in that cluster; positive can boost. Do not be given any example deltas—propose from reasoning only.
```

**User message (template with placeholders):**

```
## Cluster descriptions
{{ cluster_descriptions_json }}

## Deltas used for this minibatch (baseline to adjust from)
{{ current_deltas_json }}

## Evolving summary (what you have learnt so far; you update this each round and it is served back to you in later iterations)
{{ cot_summary }}

## All minibatch responses (with correctness and explanation per example)
{{ all_responses_with_correctness }}

---
Spot and eliminate any padding, verbosity, or fluff in these responses. Given the deltas that were used for this minibatch (above), propose updated deltas for each cluster so that responses become razor-sharp. You must also update the evolving summary: provide 2–4 sentences on what you learnt this round (e.g. which clusters added fluff, what you changed, what to try next); this update will be appended and served back to you in later iterations. Output only a JSON object with two fields: "deltas" (object with string keys "0", "1", ... and float values) and "summary" (string, your update to the evolving summary). No other commentary.
```

- `iteration_number`: current 0-based iteration index.
- `cluster_descriptions_json`: pretty-printed or compact JSON of `cluster_descriptions.json`.
- `current_deltas_json`: the deltas that were used for this minibatch (current iteration's deltas), as JSON (e.g. `{"0": -0.35, "1": -0.01, ...}`).
- `cot_summary`: the accumulated evolving summary (empty or “First iteration; no prior learnings.” on iteration 0); served back to the reflector in later iterations.
- `all_responses_with_correctness`: all minibatch responses, each with example_id, Correct (yes/no), Explanation (reasoning from correctness evaluator), and the response text.

### 4.3 Expected output from the reflector

The reflector must return **only** a JSON object with two top-level keys. No markdown, no explanation. Example:

```json
{
  "deltas": {
    "0": -0.35,
    "1": -0.01,
    "2": 0.08,
    "3": 0.12,
    "4": 0.10,
    "5": 0.10,
    "6": -0.15,
    "7": 0.08,
    "8": 0.10,
    "9": -0.12,
    "10": 0.10,
    "11": 0.10
  },
  "summary": "Cluster 6 (e.g. 'of') and cluster 9 (e.g. 'the') still produced filler; increased their negative deltas. EOS (0) kept at -0.35 to avoid early stop. Next iteration: watch for repeated list markers."
}
```

- **deltas:** object with string keys (cluster ids `"0"` … `"11"`) and float values (steering deltas). Negative = discourage that cluster’s tokens; positive = boost.
- **summary:** string (2–4 sentences). The reflector’s **update** to the evolving summary: what it learnt this round. The driver **appends** this to the accumulated evolving summary and **serves** it back to the reflector in later iterations.
- The driver parses the response, **validates** (see §4.4), uses `deltas` as the new deltas for the next iteration, and appends `summary` to the accumulated evolving summary (served to the reflector in later iterations).

### 4.4 OpenAI JSON schema enforcement and validation

The reflector call **must** use OpenAI’s **JSON schema enforcement** so the model returns only a valid JSON object with the required shape (no markdown, no extra text). Use the Chat Completions API with `response_format` set to a JSON schema that defines:

- A single object with **required** keys: `"deltas"` (object) and `"summary"` (string).
- For `"deltas"`: require **exactly** the expected cluster id set (same keys as in `cluster_descriptions.json` / initial deltas, e.g. `"0"` … `"11"`); each value must be a number. No extra or missing keys (use a strict schema or `additionalProperties: false` with explicit property names).
- No other top-level keys.

This ensures parseable output every time. **After** parsing, the driver **validates**: (1) `deltas` has exactly the expected cluster ids (no more, no less); (2) every value in `deltas` is a float. If validation fails, raise an error (and optionally retry or abort the iteration). Implementation: pass the schema in the request (e.g. `response_format: { type: "json_schema", json_schema: { name: "reflector_output", strict: true, schema: { ... } } }` per OpenAI’s structured outputs documentation).

---

## 5. Best Processor and Ctrl+C

- **Best criterion:** e.g. **mean token length** of minibatch responses (token length = same tokenizer as the SGLang model). Lower = less verbose = better. (Alternative: total token length; or a composite with correctness if desired later.)
- **Persistence:** each iteration write:
  - `deltas_current.json` (current iteration’s deltas).
  - If current iteration’s mean length is better than the best so far: copy `deltas_current.json` → `deltas_best.json`; optionally call `generate-processor` with `deltas_best.json` to refresh `processor_best.py`.
- **On exit (normal or Ctrl+C):** register a signal handler for SIGINT (and optionally SIGTERM). In the handler: ensure `deltas_best.json` is up to date; optionally regenerate `processor_best.py` from `deltas_best.json`; then exit. So after Ctrl+C the user can use `deltas_best.json` and `processor_best.py` from the evolution output directory.

---

## 6. Graph: Minibatch Length vs Iteration

- **Library:** use **matplotlib** (e.g. `matplotlib.pyplot`) to avoid extra heavy dependencies. Save as PNG (or SVG) in the evolution output directory.
- **Data:** keep a list of (iteration_index, mean_minibatch_token_length). Optionally also (iteration_index, total_minibatch_token_length) or per-example lengths for a box plot later.
- **Plot:** x = iteration (0, 1, 2, …), y = mean minibatch token length. Title/labels e.g. "Mean response length (tokens) vs evolution iteration". Update and save the image after each iteration (e.g. `evolution_lengths.png`).

---

## 7. File Layout (Proposed)

- **New files:**
  - `compressor_2/evolution/` (or `compressor_2/evolution.py` plus a small package):
    - Entry point: e.g. `evolution/run.py` or CLI subcommand `python -m compressor_2 evolve ...`.
    - Logic: loop, SGLang start/stop, generate-processor invocation, minibatch run (financebench_runner or in-process client), reflector call, best tracking, graph update, signal handler.
  - `compressor_2/Kconfig` (and optionally `compressor_2/defconfig`): options above.
  - `compressor_2/Makefile`: so that `make menuconfig` runs the Kconfig GUI and writes `.config`.
  - `compressor_2/kconfig_loader.py` (or equivalent): load .config into a config object for the evolution module.
- **Outputs (evolution output dir):**
  - `deltas_current.json`, `deltas_best.json`
  - `processor.py` (current), `processor_best.py` (best)
  - `evolution_lengths.png` (updated each iteration)
  - **Reflector message log:** the full message (system + user) sent to the reflector each iteration (e.g. `reflector_message_000.txt`, `reflector_message_001.txt`, … or one file with iteration markers).
  - Optional: `history.json` (iteration, mean_length, …) for reproducibility.

---

## 8. Dependencies

- Existing: `compressor_2` (embedder, pca_reducer, kmeans, logit_processor_generator), `financebench_runner` (or in-process equivalent), OpenAI client for reflector.
- New: **matplotlib** for the graph (add to `requirements.txt` if not present). Kconfig parser (e.g. `kconfiglib` as in learning_grammar) for loading .config.

---

## 9. Implementation Order (for later)

1. Add Kconfig and kconfig_loader; define options (financebench pool, minibatch size, k, iterations, paths, SGLang command, reflector model). Add a Makefile (e.g. under `compressor_2/`) so that `make menuconfig` opens the Kconfig GUI (e.g. ncurses) and writes `.config`.
2. Implement evolution loop skeleton: load config, initial deltas, cluster_descriptions; loop with iteration counter.
3. Implement SGLang lifecycle: start subprocess (configurable command), wait until healthy; stop subprocess before generate-processor.
4. Wire generate-processor (subprocess or in-process) with current deltas → processor.py; always run on GPU (no CUDA_VISIBLE_DEVICES). On GPU unavailability or OOM, raise and exit (no fallback). Ensure generate-processor implements special handling for clusters 0 and 1 (EOS/EOT and numbers/symbols) as in §2.5.
5. Implement minibatch run: call financebench_runner (or equivalent) with processor.py; collect responses and lengths; select k longest.
6. Implement reflector: build prompt (k longest + cluster_descriptions + **deltas used for this minibatch** + evolving summary); **log the full message sent to the reflector (system + user) each iteration** to the evolution output dir; maintain evolving summary across iterations (append reflector's summary update after each response and serve it back in later iterations); call OpenAI with **JSON schema enforcement** (§4.4); parse response and **validate** deltas (exact cluster ids, float values); on validation failure, raise (or retry/abort).
7. Implement best tracking: compare mean **token** length; persist deltas_best.json and processor_best.py.
8. Add matplotlib graph: update evolution_lengths.png each iteration.
9. Add SIGINT handler: flush best, optionally regenerate processor_best.py, exit.
10. Tests and docs: minimal test or dry-run; update README to describe `evolve` and the evolution output directory.
