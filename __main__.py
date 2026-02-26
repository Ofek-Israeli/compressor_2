"""
CLI for compressor_2: embed, pca, kmeans, and pipeline subcommands.
Run from the parent of this package with: python3 -m compressor_2 <subcommand> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _read_tokens(path: Path | None) -> list[str]:
    if path is None or str(path) == "-":
        return [line.rstrip("\n") for line in sys.stdin]
    return path.read_text().strip().splitlines()


def _write_lines(path: Path | None, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    if path is None or str(path) == "-":
        sys.stdout.write(text)
    else:
        path.write_text(text)


def cmd_embed(args: argparse.Namespace) -> None:
    import numpy as np
    from .embedder import embed_tokens

    tokens = _read_tokens(args.input)
    if not tokens:
        raise SystemExit("No tokens to embed (empty input).")
    X = embed_tokens(
        tokens,
        model_name=args.model,
        batch_size=getattr(args, "batch_size", None),
    )
    np.save(args.output, X)


def cmd_embed_files(args: argparse.Namespace) -> None:
    import numpy as np
    from .embedder import embed_files

    paths = [Path(p) for p in args.files]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
    X = embed_files(
        paths,
        model_name=args.model,
        batch_size=getattr(args, "batch_size", 128),
    )
    np.save(args.output, X)


def cmd_pca(args: argparse.Namespace) -> None:
    import numpy as np
    import joblib
    from .pca_reducer import reduce_pca

    X = np.load(args.input)
    Z, pca = reduce_pca(
        X,
        d=args.d,
        random_state=args.random_state,
        svd_solver=getattr(args, "svd_solver", "randomized"),
        iterated_power=getattr(args, "iterated_power", 4),
    )
    np.save(args.output, Z)
    if args.pca_out:
        joblib.dump(pca, args.pca_out)


def cmd_kmeans(args: argparse.Namespace) -> None:
    import json
    import numpy as np
    import joblib
    from .kmeans_clusterer import cluster_kmeans

    Z = np.load(args.input)
    n_init_arg = getattr(args, "n_init", "10")
    n_init = n_init_arg if n_init_arg == "auto" else int(n_init_arg)
    labels, kmeans = cluster_kmeans(
        Z,
        k=args.k,
        random_state=args.random_state,
        n_init=n_init,
    )
    _write_lines(args.output, [str(int(i)) for i in labels])

    # Model output: default path when -o is a file (not stdout)
    model_out = getattr(args, "model_out", None)
    if model_out is None and args.output is not None and str(args.output) != "-":
        model_out = args.output.with_stem(args.output.stem + "_kmeans").with_suffix(".joblib")
    if model_out is not None:
        joblib.dump(kmeans, model_out)

    # Optional: cluster descriptions via gpt-4o
    descriptions_out = getattr(args, "descriptions_out", None)
    if descriptions_out is not None:
        text_path = getattr(args, "text", None)
        if text_path is None:
            raise SystemExit("--descriptions-out requires --text (path to text file, same order as reduced).")
        try:
            from .representatives import load_text_units
            from .cluster_descriptions import describe_clusters
            text_units = load_text_units(text_path, len(labels))
            descriptions = describe_clusters(labels, Z, text_units)
            out = {str(k): v for k, v in descriptions.items()}
            descriptions_out.write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n"
            )
        except RuntimeError as e:
            raise SystemExit(str(e))


def cmd_representatives(args: argparse.Namespace) -> None:
    import json
    from .representatives import get_representatives

    result = get_representatives(
        args.embeddings,
        args.labels,
        args.reduced,
        args.text,
        args.n,
        validate_embeddings=args.validate_embeddings,
    )
    # JSON keys must be strings; use str(label_id)
    out = {str(k): v for k, v in result.items()}
    json_str = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output and str(args.output) != "-":
        args.output.write_text(json_str)
    else:
        sys.stdout.write(json_str)
        if not json_str.endswith("\n"):
            sys.stdout.write("\n")


def cmd_pipeline(args: argparse.Namespace) -> None:
    import numpy as np
    import joblib
    from . import run_pipeline

    tokens = _read_tokens(args.input)
    if not tokens:
        raise SystemExit("No tokens (empty input).")
    X, Z, labels, pca, kmeans = run_pipeline(
        tokens,
        embedding_model=args.model,
        pca_d=args.d,
        kmeans_k=args.k,
        random_state=args.random_state,
    )
    _write_lines(args.output, [str(int(i)) for i in labels])
    if args.embeddings_out:
        np.save(args.embeddings_out, X)
    if args.reduced_out:
        np.save(args.reduced_out, Z)
    if args.pca_out:
        joblib.dump(pca, args.pca_out)
    if args.kmeans_out:
        joblib.dump(kmeans, args.kmeans_out)


def cmd_generate_processor(args: argparse.Namespace) -> None:
    from .logit_processor_generator import generate_processor

    generate_processor(
        kmeans_path=str(args.kmeans),
        deltas_path=str(args.deltas),
        output_path=str(args.output),
        llm_tokenizer=args.tokenizer,
        embedding_model=args.embedding_model,
        pca_path=str(args.pca) if args.pca else None,
        embeddings_path=str(args.embeddings) if args.embeddings else None,
        pca_random_state=args.pca_random_state,
        batch_size=args.batch_size,
    )


def cmd_evolve(args: argparse.Namespace) -> None:
    import logging
    from .kconfig_loader import load_config
    from .evolution import run_evolution

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    run_evolution(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compressor_2",
        description="Embed tokens, reduce with PCA, cluster with k-means.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # embed
    p_embed = subparsers.add_parser("embed", help="Embed tokens to vectors")
    p_embed.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Input file: one token per line (default: stdin)",
    )
    p_embed.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output .npy file for embedding matrix",
    )
    p_embed.add_argument(
        "-m", "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    p_embed.add_argument(
        "-b", "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="Encode batch size for GPU utilization (default: 128)",
    )
    p_embed.set_defaults(func=cmd_embed)

    # embed-files: multiple .txt files → one embedding per token (whitespace-split)
    p_embed_files = subparsers.add_parser(
        "embed-files",
        help="Embed text from files (one vector per token, whitespace-split)",
    )
    p_embed_files.add_argument(
        "files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Input .txt (or any text) files",
    )
    p_embed_files.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output .npy file for embedding matrix",
    )
    p_embed_files.add_argument(
        "-m", "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    p_embed_files.add_argument(
        "-b", "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="Encode batch size for GPU utilization (default: 128)",
    )
    p_embed_files.set_defaults(func=cmd_embed_files)

    # pca
    p_pca = subparsers.add_parser("pca", help="Reduce embeddings to d dimensions with PCA")
    p_pca.add_argument(
        "input",
        type=Path,
        help="Input .npy embedding matrix",
    )
    p_pca.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output .npy file for reduced matrix",
    )
    p_pca.add_argument(
        "-d",
        type=int,
        required=True,
        metavar="DIM",
        help="Number of PCA components",
    )
    p_pca.add_argument(
        "--pca-out",
        type=Path,
        default=None,
        help="Optional: save fitted PCA (joblib)",
    )
    p_pca.add_argument("--random-state", type=int, default=None, help="Random seed")
    p_pca.add_argument(
        "--svd-solver",
        type=str,
        default="randomized",
        choices=("auto", "full", "arpack", "randomized"),
        help="PCA SVD solver (default: randomized for speed)",
    )
    p_pca.add_argument(
        "--iterated-power",
        type=int,
        default=4,
        metavar="N",
        help="Power iterations for randomized solver (default: 4)",
    )
    p_pca.set_defaults(func=cmd_pca)

    # kmeans
    p_kmeans = subparsers.add_parser("kmeans", help="Cluster reduced vectors with k-means")
    p_kmeans.add_argument(
        "input",
        type=Path,
        help="Input .npy reduced matrix",
    )
    p_kmeans.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output file: one label per line (default: stdout)",
    )
    p_kmeans.add_argument(
        "-k",
        type=int,
        required=True,
        metavar="K",
        help="Number of clusters",
    )
    p_kmeans.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Save fitted KMeans (joblib) for predict(). Default: stem_kmeans.joblib when -o is a file.",
    )
    p_kmeans.add_argument("--random-state", type=int, default=None, help="Random seed")
    p_kmeans.add_argument(
        "--n-init",
        type=str,
        default="10",
        metavar="N",
        help="K-means runs with different seeds (default: 10); use 'auto' for sklearn default",
    )
    p_kmeans.add_argument(
        "--text",
        type=Path,
        default=None,
        help="Text file (whitespace-split tokens, same order/length as reduced). Required for --descriptions-out.",
    )
    p_kmeans.add_argument(
        "--descriptions-out",
        type=Path,
        default=None,
        help="Write cluster descriptions (JSON) via gpt-4o. Requires --text and OPENAI_API_KEY.",
    )
    p_kmeans.set_defaults(func=cmd_kmeans)

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run embed → PCA → k-means")
    p_pipe.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Input file: one token per line (default: stdin)",
    )
    p_pipe.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file: one cluster label per line",
    )
    p_pipe.add_argument(
        "-m", "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    p_pipe.add_argument(
        "-d",
        type=int,
        required=True,
        metavar="DIM",
        help="PCA dimension",
    )
    p_pipe.add_argument(
        "-k",
        type=int,
        required=True,
        metavar="K",
        help="Number of k-means clusters",
    )
    p_pipe.add_argument("--embeddings-out", type=Path, default=None, help="Save embeddings .npy")
    p_pipe.add_argument("--reduced-out", type=Path, default=None, help="Save reduced .npy")
    p_pipe.add_argument("--pca-out", type=Path, default=None, help="Save fitted PCA (joblib)")
    p_pipe.add_argument("--kmeans-out", type=Path, default=None, help="Save fitted KMeans (joblib)")
    p_pipe.add_argument("--random-state", type=int, default=None, help="Random seed")
    p_pipe.set_defaults(func=cmd_pipeline)

    # representatives
    p_repr = subparsers.add_parser(
        "representatives",
        help="Get n closest representatives per cluster from embeddings, labels, reduced, and text file",
    )
    p_repr.add_argument(
        "embeddings",
        type=Path,
        help="Path to embeddings .npy",
    )
    p_repr.add_argument(
        "labels",
        type=Path,
        help="Path to labels .txt (one label per line)",
    )
    p_repr.add_argument(
        "reduced",
        type=Path,
        help="Path to reduced .npy",
    )
    p_repr.add_argument(
        "text",
        type=Path,
        help="Path to original text file (whitespace-split tokens, same order as pipeline)",
    )
    p_repr.add_argument(
        "-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of representatives per cluster (default: 10)",
    )
    p_repr.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout). Use - for stdout.",
    )
    p_repr.add_argument(
        "--no-validate-embeddings",
        action="store_false",
        dest="validate_embeddings",
        help="Skip loading embeddings for shape validation (faster)",
    )
    p_repr.set_defaults(validate_embeddings=True, func=cmd_representatives)

    # generate-processor
    p_gen = subparsers.add_parser(
        "generate-processor",
        help="Generate a standalone SGLang CustomLogitProcessor .py from KMeans + deltas",
    )
    p_gen.add_argument(
        "kmeans",
        type=Path,
        help="Fitted KMeans model (.joblib)",
    )
    p_gen.add_argument(
        "deltas",
        type=Path,
        help="Deltas JSON mapping cluster id (str) → delta float",
    )
    p_gen.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output .py file for the generated logit processor",
    )
    p_gen.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        metavar="MODEL",
        help="HuggingFace LLM tokenizer name (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    p_gen.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model used in the pipeline (default: all-MiniLM-L6-v2)",
    )
    p_gen.add_argument(
        "--pca",
        type=Path,
        default=None,
        help="Fitted PCA model (.joblib). If omitted, --embeddings is required to re-fit.",
    )
    p_gen.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Original embeddings .npy (used to re-fit PCA when --pca is not provided)",
    )
    p_gen.add_argument(
        "--pca-random-state",
        type=int,
        default=None,
        help="Random state for PCA re-fitting (should match original run)",
    )
    p_gen.add_argument(
        "-b", "--batch-size",
        type=int,
        default=512,
        metavar="N",
        help="Batch size for embedding the LLM vocab (default: 512)",
    )
    p_gen.set_defaults(func=cmd_generate_processor)

    # evolve
    p_evolve = subparsers.add_parser(
        "evolve",
        help="Run the evolution loop: evolve steering deltas via reflector feedback",
    )
    p_evolve.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to .config file (from 'make menuconfig' or manual)",
    )
    p_evolve.set_defaults(func=cmd_evolve)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
