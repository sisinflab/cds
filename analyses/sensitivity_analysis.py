from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

try:
    import ioh
except ImportError:
    ioh = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CDS sensitivity analysis on selected BBOB functions.")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--func-ids", type=int, nargs="+", default=[15, 24])
    parser.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--h-values", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125, 0.0625])
    return parser.parse_args(argv)


def get_bbob_optimum(func_id: int, dimension: int) -> float:
    if ioh is None:
        raise ImportError("Package 'ioh' is required for BBOB optima.")
    problem = ioh.get_problem(
        func_id,
        instance=1,
        dimension=dimension,
        problem_class=ioh.ProblemClass.BBOB,
    )
    return float(problem.optimum.y)


def run_sensitivity(
    dimension: int,
    budget: int,
    seeds: range,
    func_ids: list[int],
    n_values: list[int],
    h_values: list[float],
) -> None:
    from benchmarking_module import build_bbob_problem, run_cds

    for func_id in func_ids:
        problem = build_bbob_problem(func_id=func_id, dim=dimension)
        optimum = get_bbob_optimum(func_id, dimension)

        print("=" * 60)
        print(f"CDS sensitivity analysis: {problem.name}")
        print(f"Dimension: {dimension} | Budget: {budget} | Seeds: {seeds.start}..{seeds.stop - 1}")
        print("=" * 60)

        tikz_data = {n_value: [] for n_value in n_values}
        for n_value in n_values:
            print(f"\n--- N = {n_value} ---")
            for h_value in h_values:
                gaps = []
                for seed in seeds:
                    history, _ = run_cds(
                        problem=problem,
                        radius=problem.radius,
                        budget=budget,
                        seed=seed,
                        step_size=h_value,
                        num_cells=n_value,
                    )
                    final_loss = history[-1, 1] if history.size > 0 else np.inf
                    gaps.append(max(0.0, final_loss - optimum))

                mean_gap = float(np.mean(gaps))
                plot_gap = max(mean_gap, 1e-4)
                tikz_data[n_value].append(f"({h_value}, {plot_gap:.4f})")
                print(f"h={h_value:<7} mean gap={mean_gap:.4f}")

        print("\nTIKZ coordinates")
        for n_value in n_values:
            coords = " ".join(tikz_data[n_value])
            print(f"% N_init={n_value}")
            print(rf"\addplot coordinates {{ {coords} }};")
            print(rf"\addlegendentry{{$N_\mathrm{{init}}={n_value}$}}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive.")
    run_sensitivity(
        dimension=args.dimension,
        budget=args.budget,
        seeds=range(args.seeds),
        func_ids=args.func_ids,
        n_values=args.n_values,
        h_values=args.h_values,
    )


if __name__ == "__main__":
    main()
