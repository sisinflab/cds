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
    cds_cells: Tuple[int, ...] = (1, 8, 32)

    # Standard Baselines
    cma_sigma_scales: Tuple[float, ...] = (0.1, 0.3, 0.5,)
    pso_swarm_sizes: Tuple[int, ...] = (30, 50, 100,)
    de_population_sizes: Tuple[int, ...] = (10,)
    de_strategies: Tuple[str, ...] = ("best1bin",)
    lshade_pop_factors: Tuple[int, ...] = (10, 20,)

    # --- NUOVI FLAG PER LE BASELINE (Tutti attivi per il run finale) ---
    include_cds: bool = True
    include_cmaes: bool = True
    include_pso: bool = True
    include_de: bool = True
    include_random: bool = True
    include_neldermead: bool = True  # Nelder-Mead nativo Scipy (senza penalty)
    include_pdfo: bool = True  # Powell tramite PDFO/BOBYQA
    include_grid_search: bool = True  # Ablation study sulla griglia fissa
    include_bads: bool = True  # Bayesian Adaptive Direct Search
    include_nomad: bool = True  # Mesh Adaptive Direct Search (PyNomad)
    include_lshade: bool = True  # Linear Pop Size Reduction (mealpy)


def _to_scalar(value: np.ndarray | float) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


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

    def mse_gradient(theta: np.ndarray) -> np.ndarray:
        error = x_design @ theta - y
        return (1.0 / len(y)) * x_design.T @ error

    return ProblemSpec(name=f"{dim}D Linear Regression", objective=mse_loss, gradient=mse_gradient, dimension=dim,
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
    return [
        build_linear_regression_problem(dim=dim),
        *[build_bbob_problem(func_id=func_id, dim=dim) for func_id in range(1, 25)],  # Rosenbrock
        build_abs_problem(dim=dim),
        build_layeb_problem(dim=dim),
    ]


# ==========================================
# RUNNERS CON RESTART LOGIC
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

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-radius / 2, radius / 2, size=problem.dimension)
        strategy = cma.CMAEvolutionStrategy(x0, sigma_scale * radius,
                                            {"maxfevals": budget - tracker.evaluations, "verbose": -9,
                                             "bounds": [-radius, radius], "seed": current_seed})
        strategy.optimize(tracker)
        if tracker.evaluations <= prev_evals + 1: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_pso(problem: ProblemSpec, radius: float, budget: int, seed: int, swarmsize: int = 50) -> Tuple[
    np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    lower_bound = -radius * np.ones(problem.dimension)
    upper_bound = radius * np.ones(problem.dimension)
    current_seed = seed

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        max_iterations = max(1, int((budget - tracker.evaluations) / swarmsize))
        pso(tracker, lower_bound, upper_bound, swarmsize=swarmsize, maxiter=max_iterations, debug=False)
        if tracker.evaluations <= prev_evals + swarmsize: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_differential_evolution(problem: ProblemSpec, radius: float, budget: int, seed: int, popsize: int = 15,
                               strategy: str = "best1bin") -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    bounds = [(-radius, radius)] * problem.dimension
    current_seed = seed

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        max_iterations = max(1, ((budget - tracker.evaluations) // (popsize * problem.dimension)) - 1)
        try:
            differential_evolution(lambda x: tracker(x), bounds, maxiter=max_iterations, popsize=popsize,
                                   strategy=strategy, seed=current_seed, polish=False)
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
        val = tracker(x)
        return float(val) if np.isfinite(val) else 1e300

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-radius, radius, size=problem.dimension).astype(float)
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
    except ImportError:
        print("  [!] pdfo not installed. Skipping. (pip install pdfo)")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed
    bounds = np.array([[-radius, radius]] * problem.dimension)

    def objective_wrapper(x: np.ndarray) -> float:
        # Hard-stop di sicurezza: se abbiamo esaurito il budget, restituiamo
        # un costo altissimo per bloccare l'algoritmo dall'andare oltre le 5000.
        if tracker.evaluations >= budget:
            return 1e300
        val = tracker(x)
        return float(val) if np.isfinite(val) else 1e300

    while tracker.evaluations < budget:
        remaining_evals = budget - tracker.evaluations

        # PDFO (BOBYQA) richiede almeno (n + 2) valutazioni per costruire il modello iniziale.
        # Se ce ne rimangono di meno, fermiamo il restart.
        if remaining_evals < problem.dimension + 2:
            break

        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-radius, radius, size=problem.dimension).astype(float)

        try:
            pdfo.pdfo(
                objective_wrapper,
                x0,
                bounds=bounds,
                options={
                    "maxfev": remaining_evals,
                    "honour_x0": True  # Spegne il primo warning
                }
            )
        except Exception:
            pass

        # Se l'algoritmo non ha consumato nuove valutazioni (è bloccato), usciamo dal ciclo
        if tracker.evaluations <= prev_evals + 1:
            break

        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_bads_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        from pybads import BADS
    except ImportError:
        print("  [!] pybads not installed. Skipping. (pip install pybads)")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed
    lb = -radius * np.ones(problem.dimension)
    ub = radius * np.ones(problem.dimension)

    # Plausible bounds
    plb = -0.9 * radius * np.ones(problem.dimension)
    pub = 0.9 * radius * np.ones(problem.dimension)

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)
        x0 = np.random.uniform(-radius / 2, radius / 2, size=problem.dimension)
        try:
            bads = BADS(lambda x: tracker(x), x0, lb, ub, plb, pub,
                        options={"max_fun_evals": budget - tracker.evaluations, "display": "off"})
            bads.optimize()
        except Exception:
            pass
        if tracker.evaluations <= prev_evals + 1: break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_grid_search_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int, step_size: float) -> Tuple[
    np.ndarray, float]:
    """Pure Grid Search per smontare l'obiezione del Revisore 1 sull'utilità delle dinamiche cellulari."""
    tracker = ObjectiveTracker(problem.objective)
    np.random.seed(seed)
    max_c = int(np.floor(radius / step_size))

    while tracker.evaluations < budget:
        c = np.random.randint(-max_c, max_c + 1, size=problem.dimension)
        x = c * step_size
        if np.linalg.norm(x) <= radius:
            tracker(x)

    return tracker.history_array(), tracker.elapsed_time()


def run_lshade_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int, pop_factor: int = 10) -> Tuple[np.ndarray, float]:
    try:
        from mealpy.evolutionary_based import SHADE
        from mealpy import FloatVar
    except ImportError:
        print("  [!] mealpy non installato. Esegui: pip install mealpy")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    # Sintassi moderna Mealpy 3.x per i vincoli (FloatVar)
    lb = [-radius] * problem.dimension
    ub = [radius] * problem.dimension

    def objective_wrapper(x):
        val = tracker(x)
        return float(val) if np.isfinite(val) else 1e300

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations

        problem_dict = {
            "bounds": FloatVar(lb=lb, ub=ub, name="vars"),
            "minmax": "min",
            "obj_func": objective_wrapper,
            "log_to": None
        }

        model = SHADE.L_SHADE(epoch=1000, pop_size=pop_factor * problem.dimension)
        term_dict = {"max_fe": budget - tracker.evaluations}
        model.solve(problem_dict, seed=current_seed, termination=term_dict)

        if tracker.evaluations <= prev_evals + 1:
            break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_nomad_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        import PyNomad
    except ImportError:
        print("  [!] PyNomad non installato correttamente.")
        return np.empty((0, 2)), 0.0

    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed

    # Definiamo i limiti come liste
    lb = [-radius] * problem.dimension
    ub = [radius] * problem.dimension

    while tracker.evaluations < budget:
        prev_evals = tracker.evaluations
        np.random.seed(current_seed)

        # Punto di partenza casuale
        x0 = np.random.uniform(-radius, radius, size=problem.dimension).tolist()

        # DEFINIZIONE DELLA BLACKBOX SECONDO DOCUMENTAZIONE NOMAD 4
        def bb_func(x):
            try:
                # 1. Estraiamo le coordinate dall'oggetto EvalPoint
                dim = x.size()
                coords = np.array([x.get_coord(i) for i in range(dim)])

                # 2. Valutiamo tramite il nostro tracker
                val = tracker(coords)

                # 3. Gestione valori non finiti
                if not np.isfinite(val):
                    val = 1e300

                # 4. TRUCCO DOCUMENTAZIONE: passiamo il valore a NOMAD come stringa codificata
                x.setBBO(str(val).encode("UTF-8"))
                return 1  # 1: Successo della valutazione
            except Exception:
                return 0  # 0: Fallimento

        # Parametri NOMAD specifici
        params = [
            "BB_OUTPUT_TYPE OBJ",  # Specifica che l'output è la funzione obiettivo
            f"MAX_BB_EVAL {budget - tracker.evaluations}",
            "DISPLAY_DEGREE 0",  # Silenzioso
            f"SEED {current_seed}"
        ]

        try:
            # Firma corretta per PyNomadBBO: funzione, x0, lower bounds, upper bounds, parametri
            PyNomad.optimize(bb_func, x0, lb, ub, params)
        except Exception as e:
            # Se NOMAD lancia un'eccezione (es. per budget finito), lo gestiamo qui
            pass

        # Se non ci sono stati progressi nelle valutazioni o budget esaurito, stop restarts
        if tracker.evaluations <= prev_evals + 1 or tracker.evaluations >= budget:
            break
        current_seed += 1

    return tracker.history_array(), tracker.elapsed_time()


def run_random_search(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    np.random.seed(seed)
    for _ in range(budget):
        point = np.random.uniform(-radius, radius, size=problem.dimension)
        if np.linalg.norm(point) > radius:
            point = point * (radius / np.linalg.norm(point))
        tracker(point)
    return tracker.history_array(), tracker.elapsed_time()


def build_optimizer_configs(settings: BenchmarkSettings) -> List[OptimizerConfig]:
    configs: List[OptimizerConfig] =[]

    # 1. CELLULAR DIRECT SEARCH
    if getattr(settings, 'include_cds', True):
        for step_size, num_cells in itertools.product(settings.cds_step_sizes, settings.cds_cells):
            configs.append(OptimizerConfig(
                name=f"CDS (h={step_size}, cells={num_cells})",
                runner=run_cds,
                params={"step_size": step_size, "num_cells": num_cells, "legacy_initialization": settings.legacy_cds_initialization}
            ))

    # 2. CLASSIC POPULATION-BASED & EVOLUTIONARY
    if getattr(settings, 'include_cmaes', True):
        for sigma_scale in settings.cma_sigma_scales:
            configs.append(OptimizerConfig(name=f"CMA-ES (sigma={sigma_scale})", runner=run_cma_es, params={"sigma_scale": sigma_scale}))

    if getattr(settings, 'include_pso', True):
        for swarm_size in settings.pso_swarm_sizes:
            configs.append(OptimizerConfig(name=f"PSO (swarm={swarm_size})", runner=run_pso, params={"swarmsize": swarm_size}))

    if getattr(settings, 'include_de', True):
        for popsize, strategy in itertools.product(settings.de_population_sizes, settings.de_strategies):
            configs.append(OptimizerConfig(name=f"DE (pop={popsize}, str={strategy})", runner=run_differential_evolution, params={"popsize": popsize, "strategy": strategy}))

    # 3. RANDOM SEARCH (Baseline base)
    if getattr(settings, 'include_random', True):
        configs.append(OptimizerConfig(name="Random Search", runner=run_random_search, params={}))

    # 4. CLASSIC DIRECT SEARCH (Aggiornati con i bounds nativi come richiesto da Rev 1)
    if getattr(settings, 'include_neldermead', True):
        configs.append(OptimizerConfig(name="Nelder-Mead (SciPy)", runner=run_neldermead_baseline, params={}))

    if getattr(settings, 'include_pdfo', True):
        configs.append(OptimizerConfig(name="Powell (PDFO)", runner=run_pdfo_baseline, params={}))

    # 5. STATE-OF-THE-ART & MODERN BASELINES (Richiesti specificamente da Rev 1)
    if getattr(settings, 'include_bads', True):
        configs.append(OptimizerConfig(name="BADS (Bayesian Hybrid)", runner=run_bads_baseline, params={}))

    if getattr(settings, 'include_nomad', True):
        configs.append(OptimizerConfig(name="NOMAD (Mesh Adaptive)", runner=run_nomad_baseline, params={}))

    if getattr(settings, 'include_lshade', True):
        for factor in settings.lshade_pop_factors:
            configs.append(OptimizerConfig(
                name=f"L-SHADE (pop={factor}d)",
                runner=run_lshade_baseline,
                params={"pop_factor": factor}
            ))

    # 6. ABLATION STUDY (Per rispondere alla critica: "è solo merito della griglia?")
    if getattr(settings, 'include_grid_search', True):
        # Prendiamo lo step_size h mediano tra quelli testati da CDS
        h_mediano = sorted(settings.cds_step_sizes)[len(settings.cds_step_sizes) // 2]
        configs.append(OptimizerConfig(
            name=f"Pure Grid Search (h={h_mediano})",
            runner=run_grid_search_baseline,
            params={"step_size": h_mediano}
        ))

    return configs


# ==========================================
# RUNNER PRINCIPALE E UTILITY (Invariate)
# ==========================================

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
        budget_evaluations=1000, seeds=(1,2,), dimensions=(2,),
        cds_step_sizes=(0.5,), cds_cells=(8,)
    )
    results_df = run_full_benchmark(settings)
    return results_df


def main() -> None:
    settings = BenchmarkSettings()
    results_df = run_full_benchmark(settings)
    # results_df = run_quick_benchmark()
    create_summary_table(results_df)


if __name__ == "__main__":
    main()