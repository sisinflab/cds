from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np


DEFAULT_PROBLEMS = ("linear", "bbob-f8", "bbob-f15", "abs", "layeb")
DEFAULT_OPTIMIZERS = ("CDS", "CMA-ES", "Pure Grid Search", "Random Search", "Nelder-Mead")

STYLES = {
    "CDS": "blue, ultra thick",
    "CMA-ES": "green!60!black, thick",
    "NOMAD": "purple, thick",
    "Powell": "orange, thick",
    "DE": "magenta, thick",
    "PSO": "cyan!70!black, thick",
    "Pure Grid Search": "gray, thick, dashed",
    "Random Search": "black, thick, dashed",
    "Nelder-Mead": "brown, thick",
    "L-SHADE": "olive, thick",
    "BADS": "red!70!black, thick",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TikZ convergence curves by rerunning selected methods.")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS))
    parser.add_argument("--max-points", type=int, default=40)
    return parser.parse_args(argv)


def optimizer_family(name: str) -> str:
    for family in STYLES:
        if name.startswith(family) or family in name:
            return family
    return name


def build_problem(key: str, dimension: int):
    from benchmarking_module import (
        build_abs_problem,
        build_bbob_problem,
        build_linear_regression_problem,
        build_layeb_problem,
    )

    normalized = key.lower()
    if normalized == "linear":
        return "Linear Regression", build_linear_regression_problem(dim=dimension)
    if normalized.startswith("bbob-f"):
        fid = int(normalized.split("f", 1)[1])
        return f"BBOB F{fid}", build_bbob_problem(func_id=fid, dim=dimension)
    if normalized == "abs":
        return "ABS", build_abs_problem(dim=dimension)
    if normalized == "layeb":
        return "Layeb 1", build_layeb_problem(dim=dimension)
    raise ValueError(f"Unknown problem key: {key}")


def selected_optimizer_configs(optimizer_names: list[str]):
    from benchmarking_module import BenchmarkSettings, build_optimizer_configs

    settings = BenchmarkSettings(
        budget_evaluations=1,
        seeds=(0,),
        dimensions=(1,),
        cds_h_list=(0.25,),
        cds_n_list=(8,),
        cma_sigma_list=(0.3,),
        pso_swarm_list=(50,),
        lshade_pop_factors=(10,),
    )
    requested = {name.lower() for name in optimizer_names}
    configs = []
    for config in build_optimizer_configs(settings):
        family = optimizer_family(config.name)
        name_lower = config.name.lower()
        family_lower = family.lower()
        if any(request in name_lower or request in family_lower for request in requested):
            configs.append(config)
    return configs


def sample_history(history: np.ndarray, max_points: int) -> np.ndarray:
    if history.size == 0:
        return history
    indices = np.round(np.linspace(0, len(history) - 1, min(max_points, len(history)))).astype(int)
    return history[np.unique(indices)]


def print_curve(problem_title: str, problem, configs, budget: int, seed: int, max_points: int) -> None:
    print(f"\n% --- PLOT FOR: {problem_title} ---")
    print(r"\begin{tikzpicture}")
    print(
        rf"  \begin{{loglogaxis}}[width=0.48\textwidth, height=5cm, "
        rf"xlabel={{Function evaluations}}, ylabel={{Loss}}, title={{{problem_title}}}, "
        rf"grid=major, legend pos=outer north east, legend style={{font=\tiny}}]"
    )
    for config in configs:
        history, _ = config.runner(problem, problem.radius, budget, seed, **config.params)
        if history is None or history.size == 0:
            print(f"    % {config.name}: no history available")
            continue
        sampled = sample_history(history, max_points)
        coords = " ".join(
            f"({x:.0f}, {max(y, 1e-8):.4e})" for x, y in sampled if x > 0 and np.isfinite(y)
        )
        if not coords:
            print(f"    % {config.name}: no finite points available")
            continue
        family = optimizer_family(config.name)
        style = STYLES.get(family, "black, thick")
        print(rf"    \addplot[{style}] coordinates {{ {coords} }};")
        print(rf"    \addlegendentry{{{config.name}}}")
    print(r"  \end{loglogaxis}")
    print(r"\end{tikzpicture}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configs = selected_optimizer_configs(args.optimizers)
    if not configs:
        raise ValueError("No optimizer configuration matched --optimizers.")

    for key in args.problems:
        title, problem = build_problem(key, args.dimension)
        print_curve(title, problem, configs, args.budget, args.seed, args.max_points)


if __name__ == "__main__":
    main()
