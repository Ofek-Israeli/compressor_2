# compressor_2

Pipeline: **tokens → embeddings → PCA (d dimensions) → spherical k-means (k clusters)**.

## Install (Linux / macOS / zsh or bash)

Run from the **project directory** (where `requirements.txt` lives):

```bash
cd /path/to/compressor_2
pip install -r requirements.txt
```

Or from anywhere, using the full path to `requirements.txt`:

```bash
pip install -r /path/to/compressor_2/requirements.txt
```

To import `compressor_2`, run Python with the **parent** of this directory on `PYTHONPATH`:

```bash
cd /path/to/parent/of/compressor_2
PYTHONPATH=. python3 -c "from compressor_2 import run_pipeline"
```

## CLI

From the **parent** of the `compressor_2` directory (so the package is on `PYTHONPATH`):

```bash
cd /path/to/parent/of/compressor_2
PYTHONPATH=. python3 -m compressor_2 --help
```

**Subcommands:**

| Command        | Description                                              |
|----------------|----------------------------------------------------------|
| `embed`        | Tokens (one per line) → embeddings (.npy)                |
| `embed-files`  | Multiple .txt files → embeddings (.npy); one vector per token (whitespace-split) |
| `pca`          | Embeddings (.npy) → reduced (.npy)                       |
| `kmeans`       | Reduced (.npy) → labels (text) + fitted model (joblib); optional cluster descriptions (JSON) |
| `pipeline`     | Tokens → labels (runs all three)                          |
| `evolve`       | Run evolution (DEAP GA or zero-order) — see [Evolution](#evolution) |
| `generate-processor` | Build logit processor from deltas (for a 2-GPU pod)   |
| `generate-evolution-processors` | Batch-generate processor_*.py from deltas_*.json (2-GPU pod) |
| `add_eos_eot_cluster` | Add an EOS/EOT cluster to joblib, descriptions, deltas, and embeddings (in-place) |
| `expand_deltas` | Expand deltas_examples.json to one key per k-means cluster (fix dimension mismatch for evolution) |

**Examples:**

```bash
# One token per line; output is NumPy .npy
printf 'hello\nworld\nfoo\n' > tokens.txt
PYTHONPATH=. python3 -m compressor_2 embed tokens.txt -o embeddings.npy -m all-MiniLM-L6-v2

# Embed text files: one vector per token (whitespace-split)
PYTHONPATH=. python3 -m compressor_2 embed-files doc1.txt doc2.txt -o embeddings.npy --model all-mpnet-base-v2

PYTHONPATH=. python3 -m compressor_2 pca embeddings.npy -o reduced.npy -d 8 --random-state 42 --descriptions-out cluster_descriptions.json

# Spherical k-means with fixed k (L2-normalizes, then runs k-means)
PYTHONPATH=. python3 -m compressor_2 kmeans reduced.npy -o labels.txt -k 10 --random-state 42 --descriptions-out cluster_descriptions.json

# Auto-select k (silhouette + Davies-Bouldin + stability); omit -k
PYTHONPATH=. python3 -m compressor_2 kmeans reduced.npy -o labels.txt --k-min 5 --k-max 100 --random-state 42 --descriptions-out cluster_descriptions.json

# Writes labels.txt and labels_kmeans.joblib (fitted model for kmeans.predict). Override with --model-out.
# Use --no-spherical to disable L2-normalization (plain Euclidean k-means, old behavior).
```

**K-means with cluster descriptions (gpt-4o):**  
Use either `--text` (single file) or `--embed-files` (multiple files; same token order as the embed-files subcommand). Tokens must match the order of the reduced matrix. Requires `OPENAI_API_KEY`:

```bash
# Single text file (whitespace-split tokens)
PYTHONPATH=. python3 -m compressor_2 kmeans reduced.npy -o labels.txt -k 30 --random-state 42 --text /path/to/tokens.txt --descriptions-out cluster_descriptions.json

# Multiple files (same order as when creating embeddings with embed-files)
PYTHONPATH=. python3 -m compressor_2 kmeans reduced.npy -o labels.txt -k 30 --random-state 42 --embed-files doc1.txt doc2.txt --descriptions-out cluster_descriptions.json
```

**Adding an EOS/EOT cluster (for evolution):**  
To add a dedicated cluster for end-of-sequence / end-of-text tokens so evolution can tune its delta, run `add_eos_eot_cluster`. It updates in-place: `labels_kmeans.joblib`, `cluster_descriptions.json`, `deltas_examples.json`, and `embeddings.npy`. The new cluster ID is the next key after the max in the deltas file (e.g. `"12"` if deltas have `"0"`–`"11"`). Requires a tokenizer (e.g. the SGLang/LLM model name) and the same embedding model used for the pipeline:

```bash
PYTHONPATH=. python3 -m compressor_2.add_eos_eot_cluster \
  --kmeans outputs/labels_kmeans.joblib \
  --cluster-descriptions outputs/cluster_descriptions.json \
  --deltas outputs/deltas_examples.json \
  --embeddings outputs/embeddings.npy \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --embedding-model all-MiniLM-L6-v2
```

Optional: `--pca path/to/pca.joblib` to use a saved PCA (otherwise PCA is re-fit from `--embeddings`); `--initial-delta 0.0` for the new cluster’s starting delta.

**Deltas dimension vs k-means:** Evolution uses the number of keys in the initial deltas file as the search dimension; it must equal the k-means model's number of clusters. If you see "Deltas dimension mismatch", run `expand_deltas` to add missing keys (filled with 0.0):

```bash
PYTHONPATH=. python -m compressor_2.expand_deltas \
  --kmeans outputs/labels_kmeans.joblib \
  --deltas outputs/deltas_examples.json
```

**Full pipeline (tokens → labels in one go):**

```bash
# Fixed k
PYTHONPATH=. python3 -m compressor_2 pipeline tokens.txt -o labels.txt -d 8 -k 3 --random-state 42

# Auto-select k
PYTHONPATH=. python3 -m compressor_2 pipeline tokens.txt -o labels.txt -d 8 --k-min 5 --k-max 50 --random-state 42
```

Optional: `-b` / `--batch-size N` (default 128) on `embed` and `embed-files` for better GPU utilization. Optional outputs: `--embeddings-out`, `--reduced-out`, `--pca-out`, `--kmeans-out` (pipeline only). For `kmeans`, the fitted model is written by default to `stem_kmeans.joblib` when `-o` is a file (use `--model-out` to override); use `--text` or `--embed-files` and `--descriptions-out` to generate cluster descriptions via gpt-4o (requires `OPENAI_API_KEY`; `--text` is one file, `--embed-files` is multiple files, same order as when creating embeddings). Use `-` for stdin/stdout where supported (e.g. `embed - -o out.npy` reads tokens from stdin).

### Spherical k-means and auto-k selection

By default, `kmeans` and `pipeline` use **spherical k-means**: each embedding is L2-normalized (`x <- x / ||x||`) before clustering, so k-means effectively optimizes cosine distance. This is recommended for high-dimensional embeddings.

**Choosing k automatically.** When `-k` is omitted, the system sweeps `--k-min` to `--k-max` and evaluates each candidate with:

- **Silhouette (cosine):** Compactness vs separation (higher is better).
- **Davies-Bouldin:** Cluster overlap (lower is better).
- **Stability (Adjusted Rand Index):** Each candidate k is clustered `--n-seeds` times with different seeds; pairwise ARI measures agreement. High ARI means the clustering is reproducible.

The smallest k with high stability (ARI >= 0.8) and silhouette within 0.02 of the best is selected. The selected k and scores are printed to stderr.

For large datasets (N > 10000), `MiniBatchKMeans` is used automatically for speed.

## Evolution

Evolution runs **only** on a **2-GPU pod**: embedding/generate-processor on one GPU, SGLang (LLM) on the other, with prefetch to overlap processor generation and query evaluation. See `docs/2XGPU_pod_plan.md` for GPU pinning and `docs/genetic_evolution_plan.md` (GA) and `docs/zero_order_opt_plan.md` (zero-order) for full specs.

**Run evolution (single entry):**

```bash
PYTHONPATH=. python3 -m compressor_2 evolve --config .config
```

Configure via Kconfig (e.g. `make menuconfig` then build `.config`). The **Optimization Algorithm** choice selects:

- **`deap`** — DEAP genetic algorithm (default).
- **Zero-order:** `grid_search`, `random_search`, `spsa`, `random_direction_2pt`, `differential_evolution`, `cmaes`, `optuna_tpe`, `smac`, `tr_dfo`, `skopt` (skopt only when `CONFIG_EVAL_DETERMINISTIC=y`), or `hybrid` (global → TR-DFO).

Eval set is always a **fixed minibatch**; state (e.g. `zero_order_state.json`, `evolution_state.json`) is saved during the run and on Ctrl+C so re-running continues from the same place. Zero-order methods use a shared evaluator and `EvalContext` with prefetch; population/batch methods call `ctx.evaluate_batch(...)` for maximal overlap.

**Output graphs:** DEAP runs write **ga_fitness.png** (fitness vs generation) and **evolution_tree.png** to the evolution output directory, updated after each generation. Zero-order runs write **zero_order_fitness.png** (fitness vs evaluation index, best-so-far line), updated during the run and once at the end.

**Optional:** After `evolve`, generate processor scripts from saved deltas (when GPUs are free):

```bash
PYTHONPATH=. python3 -m compressor_2 generate-evolution-processors --output-dir outputs/evolution --config .config
```

**Dependencies:** Evolution needs `deap`; zero-order needs `scipy`. Optional per method: `cma`, `optuna`, `smac`, `ConfigSpace`, `scikit-optimize`, `pdfo`, `pybobyqa` — see `requirements.txt` comments.

## Usage (Python API)

### Full pipeline

```python
from compressor_2 import run_pipeline

tokens = ["hello", "world", "foo", "bar", "baz"]

# Fixed k (spherical k-means)
embeddings, reduced, labels, pca, kmeans = run_pipeline(
    tokens,
    embedding_model="all-MiniLM-L6-v2",
    pca_d=8,
    kmeans_k=3,
    random_state=42,
)

# Auto-select k
embeddings, reduced, labels, pca, kmeans = run_pipeline(
    tokens,
    embedding_model="all-MiniLM-L6-v2",
    pca_d=8,
    k_min=5,
    k_max=50,
    random_state=42,
)
# labels[i] is the cluster (0..k-1) for tokens[i]
```

### Step by step

```python
from compressor_2 import embed_tokens, embed_files, reduce_pca, cluster_kmeans

tokens = ["a", "b", "c", "d", "e"]
X = embed_tokens(tokens, model_name="all-MiniLM-L6-v2", batch_size=128)   # (n, embed_dim)

# Or embed from text files: one vector per token (whitespace-split)
X = embed_files(["doc1.txt", "doc2.txt"])   # (n_tokens, embed_dim)

Z, pca = reduce_pca(X, d=8, random_state=42)                # (n, d), fitted PCA

# Spherical k-means with fixed k
labels, kmeans = cluster_kmeans(Z, k=3, random_state=42)    # L2-normalizes internally

# Auto-select k
labels, kmeans = cluster_kmeans(Z, k=None, k_min=5, k_max=50, random_state=42)

# Disable spherical (plain Euclidean, old behavior)
labels, kmeans = cluster_kmeans(Z, k=3, spherical=False, random_state=42)
```

### Configurable parameters

| Step        | Parameter           | Default / example        |
|------------|---------------------|--------------------------|
| Embeddings | `model_name`        | `"all-MiniLM-L6-v2"`     |
| Embeddings | `batch_size`        | `128` (GPU utilization) |
| PCA        | `d`                 | number of components     |
| PCA        | `svd_solver`        | `"randomized"` (faster); `"full"` for exact |
| PCA        | `iterated_power`   | `4` (randomized solver)  |
| K-means    | `k`                 | number of clusters (None = auto-select) |
| K-means    | `spherical`         | `True` (L2-normalize before clustering) |
| K-means    | `k_min` / `k_max`   | `5` / `200` (auto-k search range) |
| K-means    | `n_seeds`           | `5` (stability seeds for auto-k) |
| K-means    | `n_init`            | `10` or `"auto"`         |
| Both       | `random_state`      | `None` (optional)        |

**Performance:** PCA defaults to `svd_solver="randomized"` for faster fitting on multi-core CPUs (e.g. Apple M4). Use `svd_solver="full"` for exact SVD when needed. CLI: `pca` supports `--svd-solver`, `--iterated-power`; `kmeans` supports `--n-init`. For large N (> 10000), `MiniBatchKMeans` is used automatically.

Use `pca.transform(new_embeddings)` and `kmeans.predict(normalize(new_reduced))` to process new data with the same fit (L2-normalize before predict when using spherical k-means).
