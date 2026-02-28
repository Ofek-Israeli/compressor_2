"""
DEAP-based genetic evolution driver for steering deltas.

This is the **only** evolution loop for compressor_2.  Entry point:
    PYTHONPATH=/workspace python3 -m compressor_2 evolve --config .config

See docs/genetic_evolution_plan.md for the full specification.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from deap import base, creator, tools

from .config import EvolutionConfig
from .ga_reflector import call_merge_reflector, call_mutation_reflector
from .sglang_lifecycle import SGLangServer

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: JSON / file I/O
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Delta list ↔ dict conversion (canonical order)
# ---------------------------------------------------------------------------

def _deltas_dict_to_list(d: Dict[str, float], cluster_ids: List[str]) -> List[float]:
    return [float(d[cid]) for cid in cluster_ids]


def _deltas_list_to_dict(lst: List[float], cluster_ids: List[str]) -> Dict[str, float]:
    return {cid: lst[i] for i, cid in enumerate(cluster_ids)}


# ---------------------------------------------------------------------------
# Round-robin state for mutation target cluster
# ---------------------------------------------------------------------------

class RRState:
    """Mutable round-robin state closed over by the mutate operator."""

    def __init__(self, cluster_ids: List[str], rr_idx: int = 0):
        self.cluster_ids = cluster_ids
        self.rr_idx = rr_idx

    def next_target(self) -> str:
        cid = self.cluster_ids[self.rr_idx]
        self.rr_idx = (self.rr_idx + 1) % len(self.cluster_ids)
        return cid


# ---------------------------------------------------------------------------
# Processor generation and training-set evaluation
# ---------------------------------------------------------------------------

def _generate_processor(cfg: EvolutionConfig, deltas_path: str, output_path: str) -> None:
    cmd = [
        sys.executable, "-m", "compressor_2", "generate-processor",
        cfg.kmeans_path, deltas_path,
        "-o", output_path,
        "--tokenizer", cfg.sglang_model_path,
        "--embedding-model", cfg.embedding_model,
        "--embeddings", cfg.embeddings_path,
        "-b", str(cfg.gen_batch_size),
    ]
    LOG.info("Running generate-processor …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOG.error("generate-processor stderr:\n%s", result.stderr)
        raise RuntimeError(f"generate-processor failed (exit {result.returncode})")


def _build_runner_config(
    base_config_path: str,
    indices: List[int],
    cfg: EvolutionConfig,
) -> str:
    """Create a temp YAML runner config for the given example indices."""
    import yaml

    with open(base_config_path, "r") as f:
        runner_cfg = yaml.safe_load(f)
    runner_cfg["example_indices"] = indices
    runner_cfg["concurrency"] = len(indices)
    runner_cfg["model_id"] = cfg.sglang_model_path
    if "sglang" not in runner_cfg or not isinstance(runner_cfg.get("sglang"), dict):
        runner_cfg["sglang"] = {}
    runner_cfg["sglang"]["base_url"] = f"http://localhost:{cfg.sglang_port}"
    runner_cfg["sglang"]["timeout_s"] = cfg.runner_timeout_s

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="ga_runner_")
    yaml.dump(runner_cfg, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def _run_training_set(
    runner_config_path: str,
    financebench_jsonl: str,
    processor_path: str,
) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_output = tmp.name
    cmd = [
        sys.executable, "-m", "financebench_runner",
        "--config", runner_config_path,
        "--input", financebench_jsonl,
        "--output", tmp_output,
        "--logit-processor", processor_path,
    ]
    LOG.info("Running financebench_runner …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOG.error("financebench_runner stderr:\n%s", result.stderr)
        raise RuntimeError(f"financebench_runner failed (exit {result.returncode})")
    with open(tmp_output, "r") as f:
        results = json.load(f)
    os.unlink(tmp_output)
    return results


def _run_correctness(results: List[Dict[str, Any]], cfg: EvolutionConfig) -> None:
    from .correctness_openai import evaluate_one

    for r in results:
        ev = evaluate_one(
            predicted=r.get("llm_answer", ""),
            ground_truth=[r.get("ground_truth_answer", "")],
            question=r.get("question", ""),
            model=cfg.correctness_model,
            tolerance=cfg.correctness_tolerance,
            api_key_env=cfg.openai_api_key_env,
        )
        r["is_correct"] = ev.is_correct
        r["correctness_reasoning"] = ev.reasoning


# ---------------------------------------------------------------------------
# Full fitness evaluation for one individual
# ---------------------------------------------------------------------------

def _evaluate_individual(
    ind: Any,
    cfg: EvolutionConfig,
    cluster_ids: List[str],
    training_indices: List[int],
    tokenizer: Any,
    server_holder: Dict[str, Any],
    out_dir: Path,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Generate processor → SGLang → training set → correctness → fitness.

    Returns (fitness_scalar, training_results).
    """
    deltas_dict = _deltas_list_to_dict(list(ind), cluster_ids)

    deltas_path = str(out_dir / "_eval_deltas.json")
    _save_json(deltas_dict, deltas_path)

    # Stop SGLang to free GPU for processor generation
    srv: Optional[SGLangServer] = server_holder.get("server")
    if srv is not None:
        srv.stop()
        server_holder["server"] = None

    processor_path = str(out_dir / "_eval_processor.py")
    _generate_processor(cfg, deltas_path, processor_path)

    srv = SGLangServer(cfg, processor_path)
    srv.start()
    server_holder["server"] = srv

    tmp_runner_cfg = _build_runner_config(cfg.runner_config_path, training_indices, cfg)
    try:
        results = _run_training_set(tmp_runner_cfg, cfg.financebench_jsonl, processor_path)
    finally:
        os.unlink(tmp_runner_cfg)

    if not results:
        LOG.warning("No results from training set evaluation")
        return 0.0, []

    _run_correctness(results, cfg)

    total_tok_len = 0
    num_correct = 0
    for r in results:
        answer = r.get("llm_answer", "")
        tok_len = len(tokenizer.encode(answer, add_special_tokens=False))
        r["tok_len"] = tok_len
        total_tok_len += tok_len
        if r.get("is_correct", False):
            num_correct += 1

    mean_tok_len = total_tok_len / len(results)
    correctness_ratio = num_correct / len(results)
    shortness_score = 1.0 / (1.0 + mean_tok_len / cfg.shortness_scale)
    fitness = cfg.lambda_shortness * shortness_score + cfg.lambda_correctness * correctness_ratio

    LOG.info(
        "  eval: mean_tok=%.0f  correct=%d/%d  shortness=%.3f  fitness=%.4f",
        mean_tok_len, num_correct, len(results), shortness_score, fitness,
    )
    return fitness, results


# ---------------------------------------------------------------------------
# State persistence and resume
# ---------------------------------------------------------------------------

def _save_state(
    out_dir: Path,
    generation: int,
    population: List[Any],
    best_individual: Any,
    best_fitness: Optional[float],
    rr_state: RRState,
    cluster_ids: List[str],
) -> None:
    state = {
        "generation": generation,
        "population": [
            {
                "deltas": _deltas_list_to_dict(list(ind), cluster_ids),
                "scratchpad": getattr(ind, "scratchpad", ""),
            }
            for ind in population
        ],
        "best_individual": (
            {
                "deltas": _deltas_list_to_dict(list(best_individual), cluster_ids),
                "scratchpad": getattr(best_individual, "scratchpad", ""),
            }
            if best_individual is not None
            else None
        ),
        "best_fitness": best_fitness,
        "rr_idx": rr_state.rr_idx,
    }
    _save_json(state, str(out_dir / "evolution_state.json"))


def _load_state(out_dir: Path) -> Optional[Dict[str, Any]]:
    path = out_dir / "evolution_state.json"
    if not path.exists():
        return None
    return _load_json(str(path))


def _restore_individual(data: Dict[str, Any], cluster_ids: List[str]) -> Any:
    """Restore a DEAP individual from persisted {deltas, scratchpad}."""
    delta_list = _deltas_dict_to_list(data["deltas"], cluster_ids)
    ind = creator.Individual(delta_list)
    ind.scratchpad = data.get("scratchpad", "")
    return ind


def _save_best_processor(
    best_ind: Any,
    cfg: EvolutionConfig,
    cluster_ids: List[str],
    out_dir: Path,
) -> None:
    """Save best individual's deltas and generate its processor."""
    deltas = _deltas_list_to_dict(list(best_ind), cluster_ids)
    best_deltas_path = str(out_dir / "deltas_best.json")
    _save_json(deltas, best_deltas_path)
    try:
        _generate_processor(cfg, best_deltas_path, str(out_dir / "processor_best.py"))
    except Exception:
        LOG.exception("Failed to generate best processor (non-fatal)")


# ---------------------------------------------------------------------------
# Evolution tree tracking
# ---------------------------------------------------------------------------

class EvolutionTree:
    """Lightweight genealogy tracker using the plan's stable node-ID scheme."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str]] = []

    def add_pop_node(self, gen: int, idx: int, fitness: float) -> str:
        nid = f"g{gen}_pop{idx}"
        self.nodes[nid] = {"gen": gen, "type": "pop", "fitness": fitness}
        return nid

    def add_child_pre(
        self, gen: int, child_idx: int, parent_ids: List[str],
    ) -> str:
        nid = f"g{gen}_child{child_idx}_pre"
        self.nodes[nid] = {"gen": gen, "type": "child_pre"}
        for pid in parent_ids:
            self.edges.append((pid, nid))
        return nid

    def add_child_mut(self, gen: int, child_idx: int, pre_id: str) -> str:
        nid = f"g{gen}_child{child_idx}_mut"
        self.nodes[nid] = {"gen": gen, "type": "child_mut"}
        self.edges.append((pre_id, nid))
        return nid

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}

    def to_latex(self) -> str:
        """Generate a standalone LaTeX (TikZ) source for the evolution tree."""
        if not self.nodes:
            return ""

        generations = sorted({info["gen"] for info in self.nodes.values()})

        gen_data: Dict[int, Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {
            g: {"pop": [], "child_pre": [], "child_mut": []}
            for g in generations
        }
        for nid, info in self.nodes.items():
            gen_data[info["gen"]][info["type"]].append((nid, info))

        def _pop_idx(nid: str) -> int:
            m = re.search(r"pop(\d+)", nid)
            return int(m.group(1)) if m else 0

        def _child_idx(nid: str) -> int:
            m = re.search(r"child(\d+)", nid)
            return int(m.group(1)) if m else 0

        for g in gen_data:
            gen_data[g]["pop"].sort(key=lambda t: _pop_idx(t[0]))
            gen_data[g]["child_pre"].sort(key=lambda t: _child_idx(t[0]))
            gen_data[g]["child_mut"].sort(key=lambda t: _child_idx(t[0]))

        X_GEN = 4.5
        X_CHILD = 2.0
        X_MUT = 3.5
        Y_SP = 0.9

        positions: Dict[str, Tuple[float, float]] = {}

        for g in generations:
            x_base = g * X_GEN

            for i, (nid, _info) in enumerate(gen_data[g]["pop"]):
                positions[nid] = (x_base, -i * Y_SP)

            pre_by_idx = {_child_idx(nid): (nid, inf) for nid, inf in gen_data[g]["child_pre"]}
            mut_by_idx = {_child_idx(nid): (nid, inf) for nid, inf in gen_data[g]["child_mut"]}
            all_cidx = sorted(set(pre_by_idx) | set(mut_by_idx))

            for row, cidx in enumerate(all_cidx):
                if cidx in pre_by_idx:
                    positions[pre_by_idx[cidx][0]] = (x_base + X_CHILD, -row * Y_SP)
                if cidx in mut_by_idx:
                    positions[mut_by_idx[cidx][0]] = (x_base + X_MUT, -row * Y_SP)

        def _tn(nid: str) -> str:
            return nid.replace("_", "")

        lines = [
            r"\documentclass[border=5mm]{standalone}",
            r"\usepackage{tikz}",
            r"\usetikzlibrary{arrows.meta}",
            r"\begin{document}",
            r"\begin{tikzpicture}[",
            r"  pop/.style={rectangle, rounded corners=2pt, draw=blue!70, fill=blue!15,",
            r"    minimum width=10mm, minimum height=6mm, inner sep=1pt, font=\tiny, align=center},",
            r"  pre/.style={rectangle, rounded corners=2pt, draw=green!60!black, fill=green!12,",
            r"    minimum width=8mm, minimum height=5mm, inner sep=1pt, font=\tiny, align=center},",
            r"  mut/.style={rectangle, rounded corners=2pt, draw=orange!80!black, fill=orange!15,",
            r"    minimum width=8mm, minimum height=5mm, inner sep=1pt, font=\tiny, align=center},",
            r"  edge/.style={-{Stealth[length=1.5mm]}, thin, gray!60},",
            r"  genlabel/.style={font=\footnotesize\bfseries, text=black!60}",
            r"]",
        ]

        for g in generations:
            x_base = g * X_GEN
            lines.append(
                f"  \\node[genlabel] at ({x_base + X_CHILD / 2:.2f}, 1.0) {{Gen {g}}};"
            )

            for nid, info in gen_data[g]["pop"]:
                x, y = positions[nid]
                fit = info.get("fitness", 0)
                idx = _pop_idx(nid)
                label = f"p{idx}\\\\[-1pt]{{\\scriptsize {fit:.2f}}}"
                lines.append(
                    f"  \\node[pop] ({_tn(nid)}) at ({x:.2f}, {y:.2f}) {{{label}}};"
                )

            for nid, _info in gen_data[g]["child_pre"]:
                x, y = positions[nid]
                idx = _child_idx(nid)
                lines.append(
                    f"  \\node[pre] ({_tn(nid)}) at ({x:.2f}, {y:.2f}) {{c{idx}}};"
                )

            for nid, _info in gen_data[g]["child_mut"]:
                x, y = positions[nid]
                idx = _child_idx(nid)
                lines.append(
                    f"  \\node[mut] ({_tn(nid)}) at ({x:.2f}, {y:.2f}) {{m{idx}}};"
                )

        lines.append("")
        for src, dst in self.edges:
            if src in positions and dst in positions:
                lines.append(f"  \\draw[edge] ({_tn(src)}) -- ({_tn(dst)});")

        lines.extend([r"\end{tikzpicture}", r"\end{document}"])
        return "\n".join(lines)


def _compile_tree_png(tree: EvolutionTree, out_dir: Path) -> None:
    """Write evolution_tree.tex and compile to PNG via pdflatex + pdftoppm."""
    latex_src = tree.to_latex()
    if not latex_src:
        return

    tex_path = out_dir / "evolution_tree.tex"
    tex_path.write_text(latex_src)
    LOG.info("Wrote %s", tex_path)

    if shutil.which("pdflatex") is None:
        LOG.warning("pdflatex not found; skipping tree PNG compilation")
        return

    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "evolution_tree.tex"],
            cwd=str(out_dir),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            LOG.warning("pdflatex failed (exit %d); check evolution_tree.log",
                        result.returncode)
            return

        pdf_path = out_dir / "evolution_tree.pdf"
        png_base = str(out_dir / "evolution_tree")

        if shutil.which("pdftoppm") is None:
            LOG.warning("pdftoppm not found; PDF created but no PNG")
            return

        result = subprocess.run(
            ["pdftoppm", "-png", "-r", "150", "-singlefile",
             str(pdf_path), png_base],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            LOG.warning("pdftoppm failed (exit %d)", result.returncode)
            return

        LOG.info("Tree PNG updated: %s", out_dir / "evolution_tree.png")

    except subprocess.TimeoutExpired:
        LOG.warning("Tree compilation timed out")
    except Exception:
        LOG.exception("Error compiling tree PNG (non-fatal)")
    finally:
        for ext in (".aux", ".log"):
            p = out_dir / f"evolution_tree{ext}"
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# Main GA evolution loop
# ---------------------------------------------------------------------------

def run_ga_evolution(cfg: EvolutionConfig) -> None:
    """DEAP-based genetic evolution of steering deltas (the only evolution loop)."""

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    cluster_descriptions: Dict[str, Any] = _load_json(cfg.cluster_descriptions_path)
    initial_deltas: Dict[str, float] = _load_json(cfg.initial_deltas_path)
    cluster_ids = sorted(initial_deltas.keys(), key=lambda x: int(x))
    M = len(cluster_ids)
    cfg.expected_cluster_ids = cluster_ids
    LOG.info("Cluster IDs (%d): %s", M, cluster_ids)

    # ---- Training set (fixed for the whole run) ----
    from financebench_runner.data import load_financebench

    all_examples = load_financebench(cfg.financebench_jsonl)
    if cfg.pool_indices is not None:
        training_indices = [i for i in cfg.pool_indices if 0 <= i < len(all_examples)]
    else:
        training_indices = list(range(len(all_examples)))
    LOG.info("Training set: %d examples (fixed)", len(training_indices))

    # ---- Tokenizer ----
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.sglang_model_path)

    # ---- DEAP types ----
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax, scratchpad="")

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)

    # ---- Round-robin state ----
    rr_state = RRState(cluster_ids, rr_idx=0)

    # ---- Server state (mutable, shared across evaluations) ----
    server_holder: Dict[str, Any] = {"server": None}

    # ---- Evolution tree ----
    tree = EvolutionTree()

    # ---- Resume or fresh start ----
    saved = _load_state(out_dir)
    generation_start = 0
    best_individual: Any = None
    best_fitness: Optional[float] = None
    gen_history: List[Dict[str, Any]] = []

    if saved is not None:
        generation_start = saved["generation"] + 1
        rr_state.rr_idx = saved.get("rr_idx", 0)
        population = [
            _restore_individual(d, cluster_ids) for d in saved["population"]
        ]
        if saved.get("best_individual") is not None:
            best_individual = _restore_individual(saved["best_individual"], cluster_ids)
            best_fitness = saved.get("best_fitness")
        ga_history_path = out_dir / "ga_history.json"
        if ga_history_path.exists():
            try:
                loaded = _load_json(str(ga_history_path))
                if isinstance(loaded, list) and loaded:
                    gen_history.extend(loaded)
                    LOG.info("Loaded %d prior generations from ga_history.json",
                             len(gen_history))
            except Exception:
                LOG.warning("Could not load ga_history.json; starting with empty history")
        LOG.info(
            "Resumed from generation %d (rr_idx=%d, best_fitness=%s)",
            saved["generation"], rr_state.rr_idx,
            f"{best_fitness:.4f}" if best_fitness is not None else "N/A",
        )
    else:
        initial_list = _deltas_dict_to_list(initial_deltas, cluster_ids)
        population = []
        for _ in range(cfg.population_size):
            ind = creator.Individual(list(initial_list))
            ind.scratchpad = ""
            del ind.fitness.values
            population.append(ind)
        LOG.info("Created initial population: %d clones of initial deltas", cfg.population_size)

    # ---- SIGINT handler ----
    interrupted = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        LOG.warning("SIGINT received — persisting state and exiting after current step …")

    signal.signal(signal.SIGINT, _sigint_handler)

    last_gen = generation_start  # track for final persist

    try:
        for gen in range(generation_start, cfg.ngen):
            if interrupted:
                break
            last_gen = gen
            LOG.info("========== Generation %d / %d ==========", gen, cfg.ngen - 1)

            # ---- (1) Evaluate all individuals that need it ----
            for i, ind in enumerate(population):
                if not ind.fitness.valid:
                    LOG.info("Evaluating pop[%d] …", i)
                    fit, results = _evaluate_individual(
                        ind, cfg, cluster_ids, training_indices,
                        tokenizer, server_holder, out_dir,
                    )
                    ind.fitness.values = (fit,)
                    ind.training_results = results
                tree.add_pop_node(gen, i, ind.fitness.values[0])

            # ---- Elitism: select elites NOW (before offspring) ----
            elites = tools.selBest(population, cfg.elitism_size)
            elite_clones = [toolbox.clone(e) for e in elites]

            # ---- Update best-ever ----
            for ind in population:
                f = ind.fitness.values[0]
                if best_fitness is None or f > best_fitness:
                    best_fitness = f
                    best_individual = toolbox.clone(ind)

            pop_fits = [ind.fitness.values[0] for ind in population]
            LOG.info(
                "Gen %d pop: best=%.4f  mean=%.4f  best_ever=%.4f",
                gen, max(pop_fits), sum(pop_fits) / len(pop_fits),
                best_fitness if best_fitness is not None else 0,
            )

            # ---- (2) Selection ----
            n_offspring = cfg.population_size - cfg.elitism_size
            n_parents = 2 * n_offspring

            if cfg.selection == "truncation":
                parent_pool = tools.selBest(population, cfg.truncation_top_k)
                parents = [random.choice(parent_pool) for _ in range(n_parents)]
            else:  # tournament
                parents = tools.selTournament(population, n_parents, cfg.tournament_size)

            # ---- (3) Crossover → build full offspring list ----
            offspring: List[Any] = []
            for j in range(n_offspring):
                p1 = parents[2 * j]
                p2 = parents[2 * j + 1]
                p1_idx = population.index(p1)
                p2_idx = population.index(p2)

                if random.random() < cfg.cxpb:
                    child = _mate(
                        toolbox.clone(p1), toolbox.clone(p2),
                        cfg, cluster_ids, cluster_descriptions,
                        output_dir=str(out_dir), generation=gen, child_idx=j,
                    )
                else:
                    chosen = random.choice([p1, p2])
                    child = toolbox.clone(chosen)
                    p2_idx = p1_idx  # single parent for tree

                del child.fitness.values

                parent_nids = [f"g{gen}_pop{p1_idx}"]
                if p1_idx != p2_idx:
                    parent_nids.append(f"g{gen}_pop{p2_idx}")
                tree.add_child_pre(gen, j, parent_nids)

                offspring.append(child)

            # ---- (4) For each child: evaluate, maybe mutate + re-evaluate ----
            for j, child in enumerate(offspring):
                if interrupted:
                    break

                LOG.info("Evaluating offspring[%d] …", j)
                fit, results = _evaluate_individual(
                    child, cfg, cluster_ids, training_indices,
                    tokenizer, server_holder, out_dir,
                )
                child.fitness.values = (fit,)
                child.training_results = results

                if random.random() < cfg.mutpb:
                    LOG.info("Mutating offspring[%d] …", j)
                    pre_nid = f"g{gen}_child{j}_pre"

                    c_target = rr_state.next_target()
                    deltas_dict = _deltas_list_to_dict(list(child), cluster_ids)

                    mutation_result = call_mutation_reflector(
                        cfg, cluster_descriptions, deltas_dict,
                        child.scratchpad, c_target, results,
                        output_dir=str(out_dir), generation=gen, index=j,
                    )

                    if mutation_result is not None:
                        new_delta, new_scratchpad = mutation_result
                        target_idx = cluster_ids.index(c_target)
                        child[target_idx] = new_delta
                        child.scratchpad = new_scratchpad
                        del child.fitness.values

                        tree.add_child_mut(gen, j, pre_nid)

                        LOG.info("Re-evaluating mutated offspring[%d] …", j)
                        fit2, results2 = _evaluate_individual(
                            child, cfg, cluster_ids, training_indices,
                            tokenizer, server_holder, out_dir,
                        )
                        child.fitness.values = (fit2,)
                        child.training_results = results2

                # Update best-ever from offspring
                f = child.fitness.values[0]
                if best_fitness is None or f > best_fitness:
                    best_fitness = f
                    best_individual = toolbox.clone(child)

            # ---- (5) Replace: next population = elites + offspring ----
            population = elite_clones + offspring

            # ---- Logging / persistence ----
            all_fits = [
                ind.fitness.values[0] for ind in population if ind.fitness.valid
            ]
            gen_record = {
                "generation": gen,
                "best_fitness": max(all_fits) if all_fits else 0,
                "mean_fitness": sum(all_fits) / len(all_fits) if all_fits else 0,
                "best_ever_fitness": best_fitness,
            }
            gen_history.append(gen_record)
            LOG.info(
                "Gen %d done: pop_best=%.4f  pop_mean=%.4f  best_ever=%.4f",
                gen,
                gen_record["best_fitness"],
                gen_record["mean_fitness"],
                best_fitness if best_fitness is not None else 0,
            )

            _save_state(out_dir, gen, population, best_individual, best_fitness, rr_state, cluster_ids)
            _save_json(gen_history, str(out_dir / "ga_history.json"))
            _save_json(tree.to_dict(), str(out_dir / "evolution_tree.json"))

            if best_individual is not None:
                _save_best_processor(best_individual, cfg, cluster_ids, out_dir)

            _update_fitness_graph(gen_history, str(out_dir))

            try:
                _compile_tree_png(tree, out_dir)
            except Exception:
                LOG.exception("Tree PNG compilation failed (non-fatal)")

            if interrupted:
                break

    finally:
        signal.signal(signal.SIGINT, original_sigint)

        srv = server_holder.get("server")
        if srv is not None:
            try:
                srv.stop()
            except Exception:
                LOG.exception("Error stopping SGLang during cleanup")

        _save_state(
            out_dir, last_gen, population,
            best_individual, best_fitness, rr_state, cluster_ids,
        )
        _save_json(tree.to_dict(), str(out_dir / "evolution_tree.json"))

        try:
            _compile_tree_png(tree, out_dir)
        except Exception:
            LOG.exception("Tree PNG compilation failed during cleanup (non-fatal)")

        if best_individual is not None:
            bd = _deltas_list_to_dict(list(best_individual), cluster_ids)
            _save_json(bd, str(out_dir / "deltas_best.json"))
            LOG.info("Best-ever fitness: %.4f", best_fitness or 0)
        else:
            LOG.warning("No best individual recorded")

        LOG.info("Evolution outputs in %s", cfg.output_dir)


# ---------------------------------------------------------------------------
# Custom mate (§4): fitness-weighted average + reflector scratchpad merge
# ---------------------------------------------------------------------------

def _mate(
    ind1: Any,
    ind2: Any,
    cfg: EvolutionConfig,
    cluster_ids: List[str],
    cluster_descriptions: Dict[str, Any],
    *,
    output_dir: Optional[str] = None,
    generation: int = 0,
    child_idx: int = 0,
) -> Any:
    """Crossover: fitness-weighted average deltas + reflector scratchpad merge.

    Returns a single child with invalidated fitness.
    """
    f1 = ind1.fitness.values[0]
    f2 = ind2.fitness.values[0]
    total = f1 + f2

    child_list: List[float] = []
    if total > 0:
        for i in range(len(ind1)):
            child_list.append((f1 * ind1[i] + f2 * ind2[i]) / total)
    else:
        for i in range(len(ind1)):
            child_list.append((ind1[i] + ind2[i]) / 2.0)

    child = creator.Individual(child_list)

    merged_scratchpad = call_merge_reflector(
        cfg,
        getattr(ind1, "scratchpad", ""),
        getattr(ind2, "scratchpad", ""),
        f1,
        f2,
        output_dir=output_dir, generation=generation, index=child_idx,
    )
    child.scratchpad = merged_scratchpad

    del child.fitness.values
    return child


# ---------------------------------------------------------------------------
# Fitness graph
# ---------------------------------------------------------------------------

def _update_fitness_graph(history: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot fitness vs generation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not history:
        return

    out = Path(output_dir)
    gens = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    mean = [h["mean_fitness"] for h in history]
    best_ever = [h["best_ever_fitness"] if h["best_ever_fitness"] is not None else 0.0 for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, best, "o-", label="Pop best", color="C0")
    ax.plot(gens, mean, "s--", label="Pop mean", color="C1")
    ax.plot(gens, best_ever, "^-", label="Best ever", color="C2", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Evolution: Fitness vs Generation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(str(out / "ga_fitness.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Dedicated graph: objective (best logit processor so far) vs generation
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(gens, best_ever, "o-", color="C2", linewidth=2, markersize=6)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Objective (fitness)")
    ax2.set_title("Objective: Best logit processor so far vs generation")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    fig2.savefig(str(out / "ga_objective.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)
