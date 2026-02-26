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
