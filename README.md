# compressor_2

Pipeline: **tokens → embeddings → PCA (d dimensions) → k-means (k clusters)**.

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

**Examples:**

```bash
# One token per line; output is NumPy .npy
printf 'hello\nworld\nfoo\n' > tokens.txt
PYTHONPATH=. python3 -m compressor_2 embed tokens.txt -o embeddings.npy -m all-MiniLM-L6-v2

# Embed text files: one vector per token (whitespace-split)
PYTHONPATH=. python3 -m compressor_2 embed-files doc1.txt doc2.txt -o embeddings.npy --model all-mpnet-base-v2

PYTHONPATH=. python3 -m compressor_2 pca embeddings.npy -o reduced.npy -d 8 --random-state 42
PYTHONPATH=. python3 -m compressor_2 kmeans reduced.npy -o labels.txt -k 3 --random-state 42
# Writes labels.txt and labels_kmeans.joblib (fitted model for kmeans.predict). Override with --model-out.
```

**K-means with cluster descriptions (gpt-4o):**  
Requires a text file with whitespace-split tokens (same order as when creating embeddings), and `OPENAI_API_KEY`:

```bash
PYTHONPATH=. python3 -m compressor_2 kmeans reduced.npy -o labels.txt -k 30 --random-state 42 \
  --text /path/to/tokens.txt --descriptions-out cluster_descriptions.json
```

**Full pipeline (tokens → labels in one go):**

```bash
PYTHONPATH=. python3 -m compressor_2 pipeline tokens.txt -o labels.txt -d 8 -k 3 --random-state 42
```

Optional: `-b` / `--batch-size N` (default 128) on `embed` and `embed-files` for better GPU utilization. Optional outputs: `--embeddings-out`, `--reduced-out`, `--pca-out`, `--kmeans-out` (pipeline only). For `kmeans`, the fitted model is written by default to `stem_kmeans.joblib` when `-o` is a file (use `--model-out` to override); use `--text` and `--descriptions-out` to generate cluster descriptions via gpt-4o (requires `OPENAI_API_KEY`; text file must be whitespace-split tokens in same order as pipeline). Use `-` for stdin/stdout where supported (e.g. `embed - -o out.npy` reads tokens from stdin).

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
embeddings, reduced, labels, pca, kmeans = run_pipeline(
    tokens,
    embedding_model="all-MiniLM-L6-v2",
    pca_d=8,
    kmeans_k=3,
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
labels, kmeans = cluster_kmeans(Z, k=3, random_state=42)   # (n,), fitted KMeans
```

### Configurable parameters

| Step        | Parameter           | Default / example        |
|------------|---------------------|--------------------------|
| Embeddings | `model_name`        | `"all-MiniLM-L6-v2"`     |
| Embeddings | `batch_size`        | `128` (GPU utilization) |
| PCA        | `d`                 | number of components     |
| PCA        | `svd_solver`        | `"randomized"` (faster); `"full"` for exact |
| PCA        | `iterated_power`   | `4` (randomized solver)  |
| K-means    | `k`                 | number of clusters       |
| K-means    | `n_init`            | `10` or `"auto"`         |
| Both       | `random_state`      | `None` (optional)        |

**Performance:** PCA defaults to `svd_solver="randomized"` for faster fitting on multi-core CPUs (e.g. Apple M4). Use `svd_solver="full"` for exact SVD when needed. CLI: `pca` supports `--svd-solver`, `--iterated-power`; `kmeans` supports `--n-init`.

Use `pca.transform(new_embeddings)` and `kmeans.predict(new_reduced)` to process new data with the same fit.
