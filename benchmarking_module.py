from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cma
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyswarm import pso
from scipy.optimize import differential_evolution, minimize
from scipy.stats import ortho_group
from sklearn.preprocessing import StandardScaler

from core import ObjectiveTracker, SphereCellularDirectSearch


def _configure_matplotlib_backend() -> None:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass


_configure_matplotlib_backend()


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    objective: Callable[[np.ndarray], np.ndarray]
    dimension: int
    gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None
    true_solution: Optional[np.ndarray] = None
    radius: float = 5.0
    rotation_seed: Optional[int] = None


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    runner: Callable[..., Tuple[np.ndarray, float]]
    params: Dict[str, float | int | str]


@dataclass(frozen=True)
class BenchmarkSettings:
    budget_evaluations: int = 5000
    seeds: Tuple[int, ...] = tuple(range(20))
    dimensions: Tuple[int, ...] = (2, 6, 10)
    legacy_cds_initialization: bool = True
    cds_step_sizes: Tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
    cds_cells: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    pgd_learning_rates: Tuple[float, ...] = (0.001, 0.01, 0.05, 0.1, 0.5)
    cma_sigma_scales: Tuple[float, ...] = (0.1, 0.3, 0.5)
    pso_swarm_sizes: Tuple[int, ...] = (10, 30, 50)
    de_population_sizes: Tuple[int, ...] = (10, 20, 30)
    de_strategies: Tuple[str, ...] = ("best1bin", "rand1bin")
    include_scipy_baselines: bool = False
    scipy_methods: Tuple[str, ...] = ("Nelder-Mead", "Powell")


def _to_scalar(value: np.ndarray | float) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def build_linear_regression_problem(dim: int = 6, n_points: int = 1000) -> ProblemSpec:
    np.random.seed(dim)
    n_features = dim - 1
    x_raw = np.random.uniform(2.0, 10.0, size=(n_points, n_features))
    beta_true = np.random.randn(dim)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    x_design = np.hstack([np.ones((n_points, 1)), x_scaled])

    epsilon = np.random.normal(0.0, 1.0, n_points)
    y = x_design @ beta_true + epsilon

    def mse_loss(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        predictions = x_design @ theta_batch.T
        errors = predictions - y[:, np.newaxis]
        return np.mean(0.5 * (errors ** 2), axis=0)

    def mse_gradient(theta: np.ndarray) -> np.ndarray:
        error = x_design @ theta - y
        return (1.0 / len(y)) * x_design.T @ error

    return ProblemSpec(
        name=f"{dim}D Linear Regression",
        objective=mse_loss,
        gradient=mse_gradient,
        dimension=dim,
        true_solution=beta_true,
        radius=10.0,
    )


def build_rosenbrock_problem(dim: int = 6) -> ProblemSpec:
    def rosenbrock(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        return np.sum(
            100.0 * (theta_batch[:, 1:] - theta_batch[:, :-1] ** 2.0) ** 2.0 + (1.0 - theta_batch[:, :-1]) ** 2.0,
            axis=1,
        )

    return ProblemSpec(
        name=f"{dim}D Rosenbrock",
        objective=rosenbrock,
        dimension=dim,
        true_solution=np.ones(dim),
        radius=5.0,
    )


def build_rastrigin_problem(dim: int = 6) -> ProblemSpec:
    def rastrigin(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        return 10.0 * dim + np.sum(theta_batch ** 2 - 10.0 * np.cos(2.0 * np.pi * theta_batch), axis=1)

    return ProblemSpec(
        name=f"{dim}D Rastrigin",
        objective=rastrigin,
        dimension=dim,
        true_solution=np.zeros(dim),
        radius=5.0,
    )


def build_shifted_rotated_rastrigin_problem(
    dim: int = 10,
    seed: int = 42,
    shift_limit: float = 2.0,
    radius: float = 5.0,
) -> ProblemSpec:
    generator = np.random.default_rng(seed)
    rotation_matrix = ortho_group.rvs(dim, random_state=generator)

    shift = generator.uniform(-abs(shift_limit), abs(shift_limit), size=dim).astype(float)
    shift_norm = np.linalg.norm(shift)
    max_shift_norm = 0.8 * radius
    if shift_norm > max_shift_norm and shift_norm > 0.0:
        shift = shift * (max_shift_norm / shift_norm)

    def shifted_rotated_rastrigin(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        transformed = (theta_batch - shift[np.newaxis, :]) @ rotation_matrix.T
        return 10.0 * dim + np.sum(transformed ** 2 - 10.0 * np.cos(2.0 * np.pi * transformed), axis=1)

    return ProblemSpec(
        name=f"Shifted & Rotated {dim}D Rastrigin",
        objective=shifted_rotated_rastrigin,
        dimension=dim,
        true_solution=shift.copy(),
        radius=float(radius),
        rotation_seed=int(seed),
    )


def run_cds(
    problem: ProblemSpec,
    radius: float,
    budget: int,
    seed: int,
    step_size: float,
    num_cells: int,
    legacy_initialization: bool = True,
) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    optimizer = SphereCellularDirectSearch(
        n_variables=problem.dimension,
        radius=radius,
        step_size=step_size,
        objective_function=tracker,
        num_active_cells=num_cells,
        seed=seed,
        legacy_initialization=legacy_initialization,
    )
    optimizer.run(evaluation_budget=budget)
    return tracker.history_array(), tracker.elapsed_time()


def run_pgd(
    problem: ProblemSpec,
    radius: float,
    budget: int,
    seed: int,
    learning_rate: float,
) -> Tuple[np.ndarray, float]:
    if problem.gradient is None:
        return np.empty((0, 2), dtype=float), 0.0

    tracker = ObjectiveTracker(problem.objective)
    np.random.seed(seed)

    theta = np.random.uniform(-radius, radius, size=problem.dimension).astype(float)
    norm = np.linalg.norm(theta)
    if norm > radius:
        theta = theta * (radius / norm)

    def project(value: np.ndarray) -> np.ndarray:
        value_norm = np.linalg.norm(value)
        if value_norm <= radius:
            return value
        return value * (radius / value_norm)

    tracker(theta)

    for _ in range(1, budget):
        gradient = problem.gradient(theta)
        theta_new = project(theta - learning_rate * gradient)

        tracker.evaluations += 1
        current_loss = _to_scalar(problem.objective(theta_new))

        if current_loss < tracker.best_value:
            tracker.best_value = current_loss

        tracker.history.append((tracker.evaluations, tracker.best_value))

        if np.linalg.norm(theta_new - theta) < 1e-7:
            theta = theta_new
            break

        theta = theta_new

    final_loss = _to_scalar(problem.objective(theta))
    if final_loss < tracker.best_value:
        tracker.best_value = final_loss

    if tracker.history:
        last_evaluation = tracker.history[-1][0]
        tracker.history[-1] = (last_evaluation, tracker.best_value)

    return tracker.history_array(), tracker.elapsed_time()


def run_cma_es(
    problem: ProblemSpec,
    radius: float,
    budget: int,
    seed: int,
    sigma_scale: float = 0.5,
) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    initial_sigma = sigma_scale * radius

    strategy = cma.CMAEvolutionStrategy(
        np.zeros(problem.dimension),
        initial_sigma,
        {"maxfevals": budget, "verbose": -9, "bounds": [-radius, radius], "seed": seed},
    )
    strategy.optimize(tracker)
    return tracker.history_array(), tracker.elapsed_time()


def run_pso(
    problem: ProblemSpec,
    radius: float,
    budget: int,
    seed: int,
    swarmsize: int = 50,
) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    lower_bound = -radius * np.ones(problem.dimension)
    upper_bound = radius * np.ones(problem.dimension)

    np.random.seed(seed)
    max_iterations = max(1, int(budget / swarmsize))
    pso(tracker, lower_bound, upper_bound, swarmsize=swarmsize, maxiter=max_iterations, debug=False)

    return tracker.history_array(), tracker.elapsed_time()


def run_random_search(
    problem: ProblemSpec,
    radius: float,
    budget: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    np.random.seed(seed)

    for _ in range(budget):
        point = np.random.uniform(-radius, radius, size=problem.dimension)
        norm = np.linalg.norm(point)
        if norm > radius:
            point = point * (radius / norm)
        tracker(point)

    return tracker.history_array(), tracker.elapsed_time()


def run_differential_evolution(
    problem: ProblemSpec,
    radius: float,
    budget: int,
    seed: int,
    popsize: int = 15,
    strategy: str = "best1bin",
) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    bounds = [(-radius, radius)] * problem.dimension

    total_population = popsize * problem.dimension
    max_iterations = max(1, (budget // total_population) - 1)

    try:
        differential_evolution(
            lambda x: tracker(x),
            bounds,
            maxiter=max_iterations,
            popsize=popsize,
            strategy=strategy,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            polish=False,
            tol=0.0,
        )
    except Exception as error:
        print(f"Differential Evolution warning: {error}")

    return tracker.history_array(), tracker.elapsed_time()


def run_scipy_baseline(
    problem: ProblemSpec,
    radius: float,
    budget: int,
    seed: int,
    method_name: str,
) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    np.random.seed(seed)

    x0 = np.random.uniform(-radius, radius, size=problem.dimension).astype(float)
    norm = np.linalg.norm(x0)
    if norm > radius:
        x0 = x0 * (radius / norm)

    def objective_with_penalty(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        r = np.linalg.norm(x)

        # soft penalty fuori sfera
        penalty = 0.0
        if r > radius:
            penalty = 1e3 * (r - radius) ** 2 + 1e6

        val = tracker(x)

        # blocca non-finiti
        if not np.isfinite(val):
            return 1e300

        return float(val + penalty)

    try:
        minimize(
            objective_with_penalty,
            x0,
            method=method_name,
            options={"maxfev": int(budget), "disp": False, "adaptive": True},
            tol=1e-6,
        )
    except Exception as error:
        print(f"Scipy {method_name} warning: {error}")

    return tracker.history_array(), tracker.elapsed_time()


def build_optimizer_configs(settings: BenchmarkSettings) -> List[OptimizerConfig]:
    configs: List[OptimizerConfig] = []

    for step_size, num_cells in itertools.product(settings.cds_step_sizes, settings.cds_cells):
        configs.append(
            OptimizerConfig(
                name=f"CDS (h={step_size}, cells={num_cells})",
                runner=run_cds,
                params={
                    "step_size": step_size,
                    "num_cells": num_cells,
                    "legacy_initialization": settings.legacy_cds_initialization,
                },
            )
        )

    for learning_rate in settings.pgd_learning_rates:
        configs.append(
            OptimizerConfig(
                name=f"PGD (lr={learning_rate})",
                runner=run_pgd,
                params={"learning_rate": learning_rate},
            )
        )

    for sigma_scale in settings.cma_sigma_scales:
        configs.append(
            OptimizerConfig(
                name=f"CMA-ES (sigma={sigma_scale})",
                runner=run_cma_es,
                params={"sigma_scale": sigma_scale},
            )
        )

    for swarm_size in settings.pso_swarm_sizes:
        configs.append(
            OptimizerConfig(
                name=f"PSO (swarm={swarm_size})",
                runner=run_pso,
                params={"swarmsize": swarm_size},
            )
        )

    configs.append(OptimizerConfig(name="Random Search", runner=run_random_search, params={}))

    for method_name in settings.scipy_methods:
        configs.append(
            OptimizerConfig(
                name=f"Scipy {method_name}",
                runner=run_scipy_baseline,
                params={"method_name": method_name},
            )
        )

    for popsize, strategy in itertools.product(settings.de_population_sizes, settings.de_strategies):
        configs.append(
            OptimizerConfig(
                name=f"Differential Evolution (popsize={popsize}, strategy={strategy})",
                runner=run_differential_evolution,
                params={"popsize": popsize, "strategy": strategy},
            )
        )

    return configs


def build_problem_suite(dim: int) -> List[ProblemSpec]:
    return [
        build_linear_regression_problem(dim=dim),
        build_rosenbrock_problem(dim=dim),
        build_rastrigin_problem(dim=dim),
        build_shifted_rotated_rastrigin_problem(dim=dim, seed=0),
    ]


def run_full_benchmark(settings: Optional[BenchmarkSettings] = None) -> pd.DataFrame:
    config = settings or BenchmarkSettings()
    optimizer_configs = build_optimizer_configs(config)
    records: List[Dict[str, object]] = []

    for dim in config.dimensions:
        problems = build_problem_suite(dim)

        for problem in problems:
            print(f"\n===== RUNNING ON: {problem.name} =====")

            for optimizer in optimizer_configs:
                if optimizer.runner is run_pgd and problem.gradient is None:
                    print(f"  -> Skipping {optimizer.name} (gradient unavailable).")
                    continue

                for seed in config.seeds:
                    run_problem = problem
                    instance_seed: Optional[int] = None
                    if run_problem.name.startswith("Shifted & Rotated"):
                        instance_seed = int(seed)
                        run_problem = build_shifted_rotated_rastrigin_problem(
                            dim=problem.dimension,
                            seed=instance_seed,
                            radius=problem.radius,
                        )
                        print(f"  -> Running {optimizer.name} with seed {seed} (instance_seed={instance_seed})...")
                    else:
                        print(f"  -> Running {optimizer.name} with seed {seed}...")

                    history, elapsed_time = optimizer.runner(
                        problem=run_problem,
                        radius=run_problem.radius,
                        budget=config.budget_evaluations,
                        seed=seed,
                        **optimizer.params,
                    )

                    if history.size > 0:
                        final_loss = float(history[-1, 1])
                        total_evaluations = int(history[-1, 0])
                    else:
                        final_loss = float("inf")
                        total_evaluations = config.budget_evaluations

                    records.append(
                        {
                            "Problem": run_problem.name,
                            "Instance Seed": instance_seed,
                            "Rotation Seed": run_problem.rotation_seed,
                            "Dimension": run_problem.dimension,
                            "Optimizer": optimizer.name,
                            "Seed": seed,
                            "Final Loss": final_loss,
                            "Total Evals": total_evaluations,
                            "Time (s)": elapsed_time,
                            "History": history,
                        }
                    )

    return pd.DataFrame(records)


def plot_pareto_hyperparameters(df: pd.DataFrame) -> None:
    cds_df = df[df["Optimizer"].str.contains("CDS")].copy()
    if cds_df.empty:
        return

    def extract_params(name: str) -> Tuple[float, int]:
        h_match = re.search(r"h=([\d\.]+)", name)
        cells_match = re.search(r"cells=(\d+)", name)
        if h_match is None or cells_match is None:
            raise ValueError(f"Cannot parse CDS parameters from '{name}'")
        return float(h_match.group(1)), int(cells_match.group(1))

    cds_df[["h", "cells"]] = cds_df["Optimizer"].apply(lambda name: pd.Series(extract_params(name)))

    for problem_name in cds_df["Problem"].unique():
        subset = cds_df[cds_df["Problem"] == problem_name]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=subset,
            x="Total Evals",
            y="Final Loss",
            hue="h",
            style="cells",
            palette="viridis",
            s=100,
            alpha=0.8,
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Computational Cost (Total Evaluations)", fontsize=12)
        plt.ylabel("Solution Quality (Final Loss)", fontsize=12)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.tight_layout()
        plt.show()


def plot_results(df: pd.DataFrame, budget_evaluations: int = 5000) -> None:
    for problem_name in df["Problem"].unique():
        plt.figure(figsize=(12, 8))
        subset_df = df[df["Problem"] == problem_name]

        for optimizer_name in subset_df["Optimizer"].unique():
            optimizer_runs = subset_df[subset_df["Optimizer"] == optimizer_name]
            all_histories = [row["History"] for _, row in optimizer_runs.iterrows()]

            eval_points = np.arange(1, budget_evaluations + 1)
            interpolated = []

            for history in all_histories:
                if len(history) == 0:
                    continue
                evaluations = history[:, 0]
                losses = history[:, 1]
                interp = np.interp(eval_points, evaluations, losses, left=np.inf, right=losses[-1])
                interpolated.append(interp)

            if not interpolated:
                continue

            interpolated_matrix = np.asarray(interpolated)
            mean_losses = np.mean(interpolated_matrix, axis=0)
            std_losses = np.std(interpolated_matrix, axis=0)

            plt.plot(eval_points, mean_losses, label=optimizer_name, linewidth=2)
            plt.fill_between(eval_points, mean_losses - std_losses, mean_losses + std_losses, alpha=0.2)

        plt.xlabel("Number of Function Evaluations", fontsize=14)
        plt.ylabel("Best Loss Found", fontsize=14)
        plt.title(f"Convergence Plot on {problem_name}", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.yscale("log")
        plt.xlim(1, budget_evaluations)
        plt.tight_layout()
        plt.show()


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Problem", "Optimizer"])
        .agg(
            Mean_Final_Loss=("Final Loss", "mean"),
            Std_Final_Loss=("Final Loss", "std"),
            Mean_Evals=("Total Evals", "mean"),
        )
        .reset_index()
    )

    summary["Final Loss (mu +- sigma)"] = summary.apply(
        lambda row: f"{row['Mean_Final_Loss']:.4f} +- {row['Std_Final_Loss']:.4f}",
        axis=1,
    )
    summary = summary[["Problem", "Optimizer", "Final Loss (mu +- sigma)", "Mean_Evals"]]

    print("\n--- BENCHMARK SUMMARY TABLE ---")
    print(summary.to_string(index=False))
    return summary


def run_radius_sensitivity_ablation() -> pd.DataFrame:
    print("\n===== RUNNING RADIUS SENSITIVITY ABLATION =====")

    dim = 10
    problems = [
        build_linear_regression_problem(dim=dim),
        build_rastrigin_problem(dim=dim),
        build_rosenbrock_problem(dim=dim),
    ]

    seeds = (0, 42, 123)
    results: List[Dict[str, object]] = []

    for problem in problems:
        for seed in seeds:
            budget = 5000
            step_size = 0.5
            num_cells = 32
            radius_values = (2, 10, 50, 100, 1000, 10000)

            plt.figure(figsize=(10, 7))

            for radius in radius_values:
                label = f"R={radius}" if radius < 1000 else "No Constraint (R->inf)"
                print(f"  -> Testing {label}...")

                history, elapsed_time = run_cds(
                    problem=problem,
                    radius=radius,
                    budget=budget,
                    seed=seed,
                    step_size=step_size,
                    num_cells=num_cells,
                )

                if len(history) == 0:
                    continue

                final_loss = float(history[-1, 1])
                total_evaluations = int(history[-1, 0])

                results.append(
                    {
                        "Problem": problem.name,
                        "Configuration": label,
                        "R Value": radius,
                        "Final Loss": final_loss,
                        "Total Evals": total_evaluations,
                        "Time (s)": elapsed_time,
                    }
                )

                line_style = "--" if radius > 1000 else "-"
                line_width = 3 if radius == 10 else 2
                plt.plot(history[:, 0], history[:, 1], label=label, linestyle=line_style, linewidth=line_width)

            plt.title(f"Impact of Radius Constraint on Convergence ({dim}D)", fontsize=16)
            plt.xlabel("Function Evaluations", fontsize=14)
            plt.ylabel("Best Loss Found (Log Scale)", fontsize=14)
            plt.yscale("log")
            plt.legend(fontsize=12)
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()

    df = pd.DataFrame(results)
    summary = (
        df.groupby(["Problem", "Configuration", "R Value"])
        .agg(
            Mean_Final_Loss=("Final Loss", "mean"),
            Std_Final_Loss=("Final Loss", "std"),
            Mean_Evals=("Total Evals", "mean"),
        )
        .reset_index()
    )

    summary["Final Loss (mu +- sigma)"] = summary.apply(
        lambda row: f"{row['Mean_Final_Loss']:.4f} +- {row['Std_Final_Loss']:.4f}",
        axis=1,
    )
    summary = summary[["Problem", "Configuration", "R Value", "Final Loss (mu +- sigma)", "Mean_Evals"]]

    print("\n--- RADIUS ABLATION SUMMARY ---")
    print(summary.to_string(index=False))

    print("\n--- FULL ABLATION RESULTS ---")
    print(df.to_string(index=False))

    return df


def run_cache_exploration() -> List[Tuple[int, int, float]]:
    print("\n===== CACHE HIT RATE EXPLORATION =====")

    dimensions = (2, 5, 10, 20)
    populations = (1, 5, 10, 32)
    results: List[Tuple[int, int, float]] = []

    for dimension in dimensions:
        for population in populations:
            problem = build_rastrigin_problem(dim=dimension)
            tracker = ObjectiveTracker(problem.objective)

            optimizer = SphereCellularDirectSearch(
                n_variables=dimension,
                radius=5.0,
                step_size=0.25,
                objective_function=tracker,
                num_active_cells=population,
                seed=42,
            )

            optimizer.run(evaluation_budget=5000)

            total_cache_calls = optimizer.cache_hits + optimizer.cache_misses
            hit_rate = 100.0 * optimizer.cache_hits / total_cache_calls if total_cache_calls > 0 else 0.0

            print(
                f"Dim: {dimension:2d} | Cells: {population:2d} | "
                f"Hits: {optimizer.cache_hits:4d} / {total_cache_calls:4d} | Rate: {hit_rate:.2f}%"
            )
            results.append((dimension, population, hit_rate))

    print("\n--- SUMMARY ---")
    for dimension, population, hit_rate in results:
        print(f"D={dimension}, N={population} -> {hit_rate:.1f}%")

    return results


def run_sensitivity_analysis() -> pd.DataFrame:
    print("\n===== RUNNING SENSITIVITY ANALYSIS (R and h) =====")

    problem = build_rosenbrock_problem(dim=4)
    seed = 42
    budget = 2000

    radius_values = (2, 5, 10)
    step_sizes = (2.0, 1.0, 0.5, 0.25, 0.1)

    records: List[Dict[str, float]] = []

    for radius in radius_values:
        for step_size in step_sizes:
            print(f"  -> Testing R={radius}, h={step_size}...")
            history, _ = run_cds(
                problem=problem,
                radius=radius,
                budget=budget,
                seed=seed,
                step_size=step_size,
                num_cells=10,
            )
            final_loss = float(history[-1, 1]) if len(history) > 0 else float("inf")
            records.append({"R": radius, "h": step_size, "Final Loss": final_loss})

    df = pd.DataFrame(records)
    pivot_table = df.pivot(index="h", columns="R", values="Final Loss")

    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis_r")
    plt.title("Sensitivity of CDS to R and h on Rosenbrock (4D)", fontsize=16)
    plt.xlabel("Radius (R)", fontsize=14)
    plt.ylabel("Grid Resolution (h)", fontsize=14)
    plt.show()

    return df


def run_quick_benchmark() -> Tuple[pd.DataFrame, pd.DataFrame]:
    settings = BenchmarkSettings(
        budget_evaluations=500,
        seeds=(42,),
        dimensions=(2,),
        legacy_cds_initialization=True,
        cds_step_sizes=(0.25, 0.5),
        cds_cells=(4, 16),
        pgd_learning_rates=(0.05,),
        cma_sigma_scales=(0.3,),
        pso_swarm_sizes=(10,),
        de_population_sizes=(10,),
        de_strategies=("best1bin",),
    )
    results_df = run_full_benchmark(settings)
    summary_df = create_summary_table(results_df)
    return results_df, summary_df


def main() -> None:
    settings = BenchmarkSettings()
    results_df = run_full_benchmark(settings)
    create_summary_table(results_df)


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkSettings",
    "OptimizerConfig",
    "ProblemSpec",
    "build_linear_regression_problem",
    "build_optimizer_configs",
    "build_problem_suite",
    "build_rastrigin_problem",
    "build_rosenbrock_problem",
    "build_shifted_rotated_rastrigin_problem",
    "create_summary_table",
    "main",
    "plot_pareto_hyperparameters",
    "plot_results",
    "run_cache_exploration",
    "run_cds",
    "run_cma_es",
    "run_differential_evolution",
    "run_full_benchmark",
    "run_pgd",
    "run_pso",
    "run_quick_benchmark",
    "run_radius_sensitivity_ablation",
    "run_random_search",
    "run_scipy_baseline",
    "run_sensitivity_analysis",
]
