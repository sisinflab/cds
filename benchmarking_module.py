from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cma
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyswarm import pso
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from scipy.stats import ortho_group
from sklearn.preprocessing import StandardScaler
import ioh

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
    dimensions: Tuple[int, ...] = (10, 20, 30, 40, 50)

    # CDS Configurations
    legacy_cds_initialization: bool = True
    cds_step_sizes: Tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
    cds_cells: Tuple[int, ...] = (8, 32)

    # Standard Baselines
    cma_sigma_scales: Tuple[float, ...] = (0.3,)
    pso_swarm_sizes: Tuple[int, ...] = (30, 50, 100)
    de_population_sizes: Tuple[int, ...] = (15,)
    de_strategies: Tuple[str, ...] = ("best1bin",)
    lshade_pop_factors: Tuple[int, ...] = (10, 20)

    # --- FLAG PER LE BASELINE ---
    include_cds: bool = True
    include_cmaes: bool = True
    include_pso: bool = True
    include_de: bool = True
    include_random: bool = True
    include_neldermead: bool = True
    include_pdfo: bool = True
    include_grid_search: bool = True
    include_bads: bool = True
    include_nomad: bool = True
    include_lshade: bool = True


def _to_scalar(value: np.ndarray | float) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


# ==========================================
# PROBLEM DEFINITIONS
# ==========================================

def build_linear_regression_problem(dim: int = 6, n_points: int = 1000) -> ProblemSpec:
    np.random.seed(42)  # Fissato per riproducibilità (richiesta Rev 3)
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

    return ProblemSpec(name=f"{dim}D Linear Regression", objective=mse_loss, dimension=dim,
                       true_solution=beta_true, radius=10.0)


def build_bbob_problem(func_id: int, dim: int = 10, instance: int = 1) -> ProblemSpec:
    problem = ioh.get_problem(func_id, instance=instance, dimension=dim, problem_class=ioh.ProblemClass.BBOB)

    def bbob_objective(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        return np.array([problem(x) for x in theta_batch])

    return ProblemSpec(name=f"BBOB F{func_id}: {problem.meta_data.name} ({dim}D)", objective=bbob_objective,
                       dimension=dim, true_solution=np.array(problem.optimum.x), radius=5.0)


def build_abs_problem(dim: int = 10, shift: float = 0.5) -> ProblemSpec:
    def abs_func(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        return np.sum(np.abs(theta_batch - shift), axis=1)

    return ProblemSpec(name=f"DIRECTGOLib ABS (Non-smooth) ({dim}D)", objective=abs_func, dimension=dim,
                       true_solution=np.full(dim, shift), radius=5.0)


def build_layeb_problem(dim: int = 10) -> ProblemSpec:
    def layeb(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        clipped_theta = np.clip(theta_batch, -10.0, 10.0)
        return np.sum(10000.0 * np.abs(np.sin(np.exp(clipped_theta))), axis=1)

    return ProblemSpec(name=f"DIRECTGOLib Layeb 1 (Irregular) ({dim}D)", objective=layeb, dimension=dim,
                       true_solution=None, radius=5.0)


def build_problem_suite(dim: int) -> List[ProblemSpec]:
    # 27 Funzioni totali (BBOB + extra)
    return [
        build_linear_regression_problem(dim=dim),
        # *[build_bbob_problem(func_id=func_id, dim=dim) for func_id in range(1, 25)],
        build_abs_problem(dim=dim),
        build_layeb_problem(dim=dim),
    ]


# ==========================================
# RUNNERS CON VINCOLO SFERICO RIGOROSO E RESTART LOGIC
# ==========================================

def run_cds(problem: ProblemSpec, radius: float, budget: int, seed: int, step_size: float, num_cells: int,
            legacy_initialization: bool = True) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        optimizer = SphereCellularDirectSearch(
            n_variables=problem.dimension, radius=radius, step_size=step_size,
            objective_function=tracker, num_active_cells=num_cells,
            seed=current_seed, legacy_initialization=legacy_initialization
        )
        optimizer.run(evaluation_budget=budget - tracker.evaluations)
        if tracker.evaluations <= prev_evals + 1: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_cma_es(problem: ProblemSpec, radius: float, budget: int, seed: int, sigma_scale: float = 0.5) -> Tuple[
    np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    # Hard Barrier Wrapper
    def objective_wrapper(x):
        if np.linalg.norm(x) > radius:
            tracker.evaluations += 1
            return 1e300
        return tracker(x)

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-0.5 * radius, 0.5 * radius, size=problem.dimension)

        strategy = cma.CMAEvolutionStrategy(x0, sigma_scale * radius,
                                            {"maxfevals": budget - tracker.evaluations, "verbose": -9,
                                             "bounds": [-radius, radius], "seed": current_seed})
        strategy.optimize(objective_wrapper)
        if tracker.evaluations <= prev_evals + 1: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_pso(problem: ProblemSpec, radius: float, budget: int, seed: int, swarmsize: int = 50) -> Tuple[
    np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    lower_bound = -radius * np.ones(problem.dimension)
    upper_bound = radius * np.ones(problem.dimension)
    current_seed = seed

    # PSO supporta vincoli di ineguaglianza nativi (>= 0)
    def sphere_constraint(x):
        return radius - np.linalg.norm(x)

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        max_iterations = max(1, int((budget - tracker.evaluations) / swarmsize))
        pso(tracker, lower_bound, upper_bound, swarmsize=swarmsize, maxiter=max_iterations, debug=False,
            ieqcons=[sphere_constraint])
        if tracker.evaluations <= prev_evals + swarmsize: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_differential_evolution(problem: ProblemSpec, radius: float, budget: int, seed: int, popsize: int = 15,
                               strategy: str = "best1bin") -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    bounds = [(-radius, radius)] * problem.dimension
    current_seed = seed

    # DE supporta vincoli non-lineari tramite SciPy
    nlc = NonlinearConstraint(lambda x: np.linalg.norm(x), -np.inf, radius)

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        max_iterations = max(1, ((budget - tracker.evaluations) // (popsize * problem.dimension)) - 1)
        try:
            differential_evolution(lambda x: tracker(x), bounds, maxiter=max_iterations, popsize=popsize,
                                   strategy=strategy, seed=current_seed, polish=False, constraints=(nlc,))
        except Exception:
            pass
        if tracker.evaluations <= prev_evals + popsize: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_neldermead_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed
    bounds = [(-radius, radius)] * problem.dimension

    def objective_wrapper(x: np.ndarray) -> float:
        # Hard Barrier
        if np.linalg.norm(x) > radius:
            tracker.evaluations += 1
            return 1e300

        val = tracker(x)
        return float(val) if np.isfinite(val) else 1e300

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-0.5 * radius, 0.5 * radius, size=problem.dimension).astype(float)
        try:
            minimize(objective_wrapper, x0, method='Nelder-Mead', bounds=bounds,
                     options={"maxfev": budget - tracker.evaluations, "disp": False}, tol=1e-6)
        except Exception:
            pass
        if tracker.evaluations <= prev_evals + 1: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_pdfo_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        import pdfo
        from scipy.optimize import NonlinearConstraint
    except ImportError:
        print("  [!] pdfo non installato. Skipping.")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    # 1. Bounds nativi
    bounds = np.array([[-radius, radius]] * problem.dimension)

    # 2. VINCOLO NON-LINEARE NATIVO (L2 Sfera) come da documentazione
    # La norma del vettore x deve essere compresa tra -infinito e il raggio
    nlc = NonlinearConstraint(lambda x: np.linalg.norm(x), -np.inf, radius)

    def objective_wrapper(x: np.ndarray) -> float:
        if tracker.evaluations >= budget:
            return 1e300
        val = tracker(x)
        return float(val) if np.isfinite(val) else 1e300

    while tracker.evaluations < budget:
        remaining_evals = budget - tracker.evaluations
        if remaining_evals < problem.dimension + 2:
            break

        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-0.5 * radius, 0.5 * radius, size=problem.dimension).astype(float)

        try:
            # 3. CHIAMATA A PDFO CORRETTA!
            pdfo.pdfo(
                objective_wrapper,
                x0,
                bounds=bounds,
                constraints=[nlc],  # <-- Parametro corretto secondo doc
                options={
                    "maxfev": remaining_evals,
                    "honour_x0": True
                }
            )
        except Exception as e:
            # Togliamo il silenziatore per eventuali debug
            # print(f"  [!] PDFO Error: {e}")
            pass

        if tracker.evaluations <= prev_evals + 1:
            break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()

def run_bads_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        from pybads import BADS
    except ImportError:
        print("  [!] pybads not installed. Skipping.")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    lb = -radius * np.ones(problem.dimension)
    ub = radius * np.ones(problem.dimension)
    plb = -0.9 * radius * np.ones(problem.dimension)
    pub = 0.9 * radius * np.ones(problem.dimension)

    # Vincolo nativo BADS (restituisce True se VIOLA il vincolo)
    def hypersphere_constraint(x):
        x_2d = np.atleast_2d(x)
        return np.sum(x_2d ** 2, axis=1) > radius ** 2

    while tracker.evaluations < budget:
        remaining_evals = budget - tracker.evaluations
        if remaining_evals <= 0: break

        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-0.5 * radius, 0.5 * radius, size=problem.dimension)

        try:
            bads = BADS(
                lambda x: tracker(x), x0, lb, ub, plb, pub,
                non_box_cons=hypersphere_constraint,
                options={"max_fun_evals": remaining_evals, "display": "off"}
            )
            bads.optimize()
        except Exception:
            pass

        if tracker.evaluations <= prev_evals + 1: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_grid_search_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int, step_size: float) -> Tuple[
    np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    np.random.seed(seed)
    max_c = int(np.floor(radius / step_size))

    while tracker.evaluations < budget:
        c = np.random.randint(-max_c, max_c + 1, size=problem.dimension)
        x = c * step_size
        if np.linalg.norm(x) <= radius:  # Conserviamo il vincolo circolare nativo
            tracker(x)

    return tracker.history_array(), tracker.elapsed_time()


def run_lshade_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int, pop_factor: int = 10) -> Tuple[
    np.ndarray, float]:
    try:
        from mealpy.evolutionary_based import SHADE
        from mealpy import FloatVar
    except ImportError:
        print("  [!] mealpy non installato. Esegui: pip install mealpy")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    lb = [-radius] * problem.dimension
    ub = [radius] * problem.dimension

    def objective_wrapper(x):
        norm_x = np.linalg.norm(x)
        if norm_x > radius:
            tracker.evaluations += 1
            return 1e6 + 1e3 * (norm_x - radius) ** 2

        val = tracker(x)
        return float(val) if np.isfinite(val) else 1e300

    while tracker.evaluations < budget:
        remaining_evals = budget - tracker.evaluations
        if remaining_evals <= 0:
            break

        prev_evals = tracker.evaluations

        problem_dict = {
            "bounds": FloatVar(lb=lb, ub=ub, name="vars"),
            "minmax": "min",
            "obj_func": objective_wrapper,
            "log_to": None
        }

        # Mealpy ha cambiato il formato della termination dict nelle ultime versioni.
        # Proviamo il formato ufficiale della v3: "mode": "FE" (Function Evaluations)
        term_dict = {"max_fe": remaining_evals}

        model = SHADE.L_SHADE(epoch=1000, pop_size=pop_factor * problem.dimension)

        try:
            model.solve(problem_dict, seed=current_seed, termination=term_dict)
        except Exception as e:
            # === STAMPIAMO L'ERRORE REALE ===
            print(f"\n  [!] CRASH INTERNO L-SHADE: {e}\n")
            break

        if tracker.evaluations <= prev_evals + 1:
            break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_nomad_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        import PyNomad
    except ImportError:
        print("  [!] PyNomad non installato.")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    lb = [-radius] * problem.dimension
    ub = [radius] * problem.dimension

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-0.5 * radius, 0.5 * radius, size=problem.dimension).tolist()

        def bb_func(x):
            try:
                dim = x.size()
                coords = np.array([x.get_coord(i) for i in range(dim)])

                # Hard barrier: se è fuori dalla sfera NOMAD fallisce la valutazione
                if np.linalg.norm(coords) > radius:
                    tracker.evaluations += 1
                    return 0  # 0 = Failure per i vincoli

                val = tracker(coords)
                if not np.isfinite(val): val = 1e300
                x.setBBO(str(val).encode("UTF-8"))
                return 1  # 1 = Successo
            except Exception:
                return 0

        params = [
            "BB_OUTPUT_TYPE OBJ",
            f"MAX_BB_EVAL {budget - tracker.evaluations}",
            "DISPLAY_DEGREE 0",
            f"SEED {current_seed}"
        ]

        try:
            PyNomad.optimize(bb_func, x0, lb, ub, params)
        except Exception:
            pass

        if tracker.evaluations <= prev_evals + 1 or tracker.evaluations >= budget:
            break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_random_search(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    np.random.seed(seed)

    # Rejection sampling per garantire una distribuzione uniforme DENTRO la sfera
    while tracker.evaluations < budget:
        point = np.random.uniform(-radius, radius, size=problem.dimension)
        if np.linalg.norm(point) <= radius:
            tracker(point)

    return tracker.history_array(), tracker.elapsed_time()


# ==========================================
# CONFIGURATION BUILDER E UTILITY
# ==========================================

def build_optimizer_configs(settings: BenchmarkSettings) -> List[OptimizerConfig]:
    configs: List[OptimizerConfig] = []

    if settings.include_cds:
        for h, n in itertools.product(settings.cds_step_sizes, settings.cds_cells):
            configs.append(OptimizerConfig(name=f"CDS (h={h}, N={n})", runner=run_cds,
                                           params={"step_size": h, "num_cells": n,
                                                   "legacy_initialization": settings.legacy_cds_initialization}))

    if settings.include_cmaes:
        for sigma in settings.cma_sigma_scales:
            configs.append(
                OptimizerConfig(name=f"CMA-ES (sigma={sigma})", runner=run_cma_es, params={"sigma_scale": sigma}))

    if settings.include_pso:
        for swarm in settings.pso_swarm_sizes:
            configs.append(OptimizerConfig(name=f"PSO (swarm={swarm})", runner=run_pso, params={"swarmsize": swarm}))

    if settings.include_de:
        for popsize, strategy in itertools.product(settings.de_population_sizes, settings.de_strategies):
            configs.append(
                OptimizerConfig(name=f"DE (pop={popsize}, str={strategy})", runner=run_differential_evolution,
                                params={"popsize": popsize, "strategy": strategy}))

    if settings.include_random:
        configs.append(OptimizerConfig(name="Random Search", runner=run_random_search, params={}))

    if settings.include_neldermead:
        configs.append(OptimizerConfig(name="Nelder-Mead (SciPy)", runner=run_neldermead_baseline, params={}))

    if settings.include_pdfo:
        configs.append(OptimizerConfig(name="Powell (PDFO-COBYLA)", runner=run_pdfo_baseline, params={}))

    if settings.include_bads:
        configs.append(OptimizerConfig(name="BADS (Bayesian)", runner=run_bads_baseline, params={}))

    if settings.include_nomad:
        configs.append(OptimizerConfig(name="NOMAD", runner=run_nomad_baseline, params={}))

    if settings.include_lshade:
        for factor in settings.lshade_pop_factors:
            configs.append(OptimizerConfig(name=f"L-SHADE (pop={factor}d)", runner=run_lshade_baseline,
                                           params={"pop_factor": factor}))

    if settings.include_grid_search:
        h_mediano = sorted(settings.cds_step_sizes)[len(settings.cds_step_sizes) // 2]
        configs.append(OptimizerConfig(name=f"Pure Grid Search (h={h_mediano})", runner=run_grid_search_baseline,
                                       params={"step_size": h_mediano}))

    return configs


def run_full_benchmark(settings: Optional[BenchmarkSettings] = None) -> pd.DataFrame:
    config = settings or BenchmarkSettings()
    optimizer_configs = build_optimizer_configs(config)
    records: List[Dict[str, object]] = []

    for dim in config.dimensions:
        problems = build_problem_suite(dim)

        for problem in problems:
            print(f"\n===== RUNNING ON: {problem.name} =====")

            for optimizer in optimizer_configs:
                for seed in config.seeds:
                    print(f"  -> Running {optimizer.name} with seed {seed}...")
                    history, elapsed_time = optimizer.runner(
                        problem=problem,
                        radius=problem.radius,
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

                    records.append({
                        "Problem": problem.name,
                        "Dimension": problem.dimension,
                        "Optimizer": optimizer.name,
                        "Seed": seed,
                        "Final Loss": final_loss,
                        "Total Evals": total_evaluations,
                        "Time (s)": elapsed_time,
                        "History": history,
                    })

    return pd.DataFrame(records)


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["Problem", "Optimizer"]).agg(
        Mean_Final_Loss=("Final Loss", "mean"),
        Std_Final_Loss=("Final Loss", "std"),
        Mean_Evals=("Total Evals", "mean"),
    ).reset_index()
    summary["Final Loss (mu +- sigma)"] = summary.apply(
        lambda row: f"{row['Mean_Final_Loss']:.4f} +- {row['Std_Final_Loss']:.4f}", axis=1)
    summary = summary[["Problem", "Optimizer", "Final Loss (mu +- sigma)", "Mean_Evals"]]
    print("\n--- BENCHMARK SUMMARY TABLE ---")
    print(summary.to_string(index=False))
    return summary


def run_quick_benchmark() -> Tuple[pd.DataFrame, pd.DataFrame]:
    settings = BenchmarkSettings(
        budget_evaluations=500, seeds=(42,), dimensions=(2,),
        cds_step_sizes=(0.5,), cds_cells=(8,)
    )
    results_df = run_full_benchmark(settings)
    return results_df

# Il main_cli è stato spostato in main.py come da istruzioni precedenti!