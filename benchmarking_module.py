from __future__ import annotations
import csv
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
from sklearn.preprocessing import StandardScaler
from core import ObjectiveTracker, SphereCellularDirectSearch

@dataclass(frozen=True)
class ProblemSpec:
    name: str
    objective: Callable[[np.ndarray], np.ndarray]
    dimension: int
    radius: float = 5.0
    true_solution: Optional[np.ndarray] = None

@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    runner: Callable[..., Tuple[np.ndarray, float]]
    params: Dict[str, float | int | str]

@dataclass(frozen=True)
class BenchmarkSettings:
    budget_evaluations: int = 5000
    seeds: Tuple[int, ...] = tuple(range(20))
    dimensions: Tuple[int, ...] = (10, 30, 50)
    cds_h_list: Tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
    cds_n_list: Tuple[int, ...] = (8, 32)
    cma_sigma_list: Tuple[float, ...] = (0.1, 0.3, 0.5)
    pso_swarm_list: Tuple[int, ...] = (30, 50, 100)
    lshade_pop_factors: Tuple[int, ...] = (10, 20)
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

def build_linear_regression_problem(dim: int=6) -> ProblemSpec:
    rng = np.random.default_rng(42)
    n_points = 1000
    x_raw = rng.uniform(2.0, 10.0, size=(n_points, dim - 1))
    beta_true = rng.standard_normal(dim)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    x_design = np.hstack([np.ones((n_points, 1)), x_scaled])
    y = x_design @ beta_true + rng.normal(0.0, 1.0, n_points)

    def mse_loss(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        predictions = x_design @ theta_batch.T
        errors = predictions - y[:, np.newaxis]
        return np.mean(0.5 * errors ** 2, axis=0)
    return ProblemSpec(name=f'{dim}D Linear Regression', objective=mse_loss, dimension=dim, radius=10.0)

def build_bbob_problem(func_id: int, dim: int=10) -> ProblemSpec:
    try:
        import ioh
    except ImportError as exc:
        raise ImportError("Package 'ioh' is required for BBOB problems.") from exc
    problem = ioh.get_problem(func_id, instance=1, dimension=dim, problem_class=ioh.ProblemClass.BBOB)

    def bbob_objective(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        return np.array([problem(x) for x in theta_batch])
    return ProblemSpec(name=f'BBOB F{func_id}: {problem.meta_data.name} ({dim}D)', objective=bbob_objective, dimension=dim, radius=5.0)

def build_abs_problem(dim: int=10) -> ProblemSpec:

    def abs_func(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        return np.sum(np.abs(theta_batch - 0.5), axis=1)
    return ProblemSpec(name=f'DIRECTGOLib ABS (Non-smooth) ({dim}D)', objective=abs_func, dimension=dim, radius=5.0)

def build_layeb_problem(dim: int=10) -> ProblemSpec:

    def layeb(theta: np.ndarray) -> np.ndarray:
        theta_batch = np.atleast_2d(theta)
        clipped = np.clip(theta_batch, -10.0, 10.0)
        return np.sum(10000.0 * np.abs(np.sin(np.exp(clipped))), axis=1)
    return ProblemSpec(name=f'DIRECTGOLib Layeb 1 (Irregular) ({dim}D)', objective=layeb, dimension=dim, radius=5.0)

def build_problem_suite(dim: int) -> List[ProblemSpec]:
    return [build_linear_regression_problem(dim=dim), *[build_bbob_problem(func_id=fid, dim=dim) for fid in range(1, 25)], build_abs_problem(dim=dim), build_layeb_problem(dim=dim)]

def run_cds(problem: ProblemSpec, radius: float, budget: int, seed: int, step_size: float, num_cells: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        optimizer = SphereCellularDirectSearch(problem.dimension, radius, step_size, tracker, num_cells, current_seed)
        optimizer.run(evaluation_budget=budget - tracker.evaluations)
        if tracker.evaluations <= prev + 1:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_cma_es(problem: ProblemSpec, radius: float, budget: int, seed: int, sigma_scale: float) -> Tuple[np.ndarray, float]:
    try:
        import cma
    except ImportError:
        return (np.empty((0, 2)), 0.0)
    tracker = ObjectiveTracker(problem.objective)

    def wrapper(x):
        if np.linalg.norm(x) > radius:
            tracker.evaluations += 1
            return 1e+300
        return tracker(x)
    current_seed = seed
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        np.random.seed(current_seed)
        safe_r = radius / np.sqrt(problem.dimension) * 0.9
        x0 = np.random.uniform(-safe_r, safe_r, size=problem.dimension)
        strategy = cma.CMAEvolutionStrategy(x0, sigma_scale * radius, {'maxfevals': budget - tracker.evaluations, 'verbose': -9, 'bounds': [-radius, radius], 'seed': current_seed})
        strategy.optimize(wrapper)
        if tracker.evaluations <= prev + 1:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_pso(problem: ProblemSpec, radius: float, budget: int, seed: int, swarmsize: int) -> Tuple[np.ndarray, float]:
    try:
        from pyswarm import pso
    except ImportError:
        return (np.empty((0, 2)), 0.0)
    tracker = ObjectiveTracker(problem.objective)

    def sphere_con(x):
        return radius - np.linalg.norm(x)
    current_seed = seed
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        np.random.seed(current_seed)
        max_iterations = max(1, int((budget - tracker.evaluations) / swarmsize))
        pso(tracker, -radius * np.ones(problem.dimension), radius * np.ones(problem.dimension), swarmsize=swarmsize, maxiter=max_iterations, ieqcons=[sphere_con], debug=False)
        if tracker.evaluations <= prev + swarmsize:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_differential_evolution(problem: ProblemSpec, radius: float, budget: int, seed: int, popsize: int=15) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    nlc = NonlinearConstraint(lambda x: np.linalg.norm(x), -np.inf, radius)
    current_seed = seed
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        max_iterations = max(1, (budget - tracker.evaluations) // (popsize * problem.dimension) - 1)
        try:
            differential_evolution(tracker, [(-radius, radius)] * problem.dimension, maxiter=max_iterations, popsize=popsize, seed=current_seed, polish=False, constraints=(nlc,))
        except Exception:
            pass
        if tracker.evaluations <= prev + popsize:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_neldermead_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed
    bounds = [(-radius, radius)] * problem.dimension

    def objective_wrapper(x: np.ndarray) -> float:
        if np.linalg.norm(x) > radius:
            tracker.evaluations += 1
            return 1e+300
        value = tracker(x)
        return float(value) if np.isfinite(value) else 1e+300
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        np.random.seed(current_seed)
        safe_r = radius / np.sqrt(problem.dimension) * 0.95
        x0 = np.random.uniform(-safe_r, safe_r, size=problem.dimension).astype(float)
        try:
            minimize(objective_wrapper, x0, method='Nelder-Mead', bounds=bounds, options={'maxfev': budget - tracker.evaluations, 'disp': False}, tol=1e-06)
        except Exception:
            pass
        if tracker.evaluations <= prev + 1:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_pdfo_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        import pdfo
    except ImportError:
        return (np.empty((0, 2)), 0.0)
    tracker = ObjectiveTracker(problem.objective)
    nlc = NonlinearConstraint(lambda x: np.linalg.norm(x), -np.inf, radius)
    current_seed = seed
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        safe_r = radius / np.sqrt(problem.dimension) * 0.9
        x0 = np.random.default_rng(current_seed).uniform(-safe_r, safe_r, size=problem.dimension)
        pdfo.pdfo(tracker, x0, bounds=np.array([[-radius, radius]] * problem.dimension), constraints=[nlc], options={'maxfev': budget - tracker.evaluations})
        if tracker.evaluations <= prev + 1:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_bads_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        from pybads import BADS
    except ImportError:
        return (np.empty((0, 2)), 0.0)
    tracker = ObjectiveTracker(problem.objective)

    def con(x):
        return np.sum(np.atleast_2d(x) ** 2, axis=1) > radius ** 2

    def safe_obj(x):
        if np.linalg.norm(x) > radius:
            tracker.evaluations += 1
            return 100000.0
        val = tracker(x)
        return float(val) if np.isfinite(val) else 100000.0
    current_seed = seed
    safe_r = radius / np.sqrt(problem.dimension) * 0.9
    while tracker.evaluations < budget:
        remaining = budget - tracker.evaluations
        if remaining < problem.dimension + 2:
            break
        prev = tracker.evaluations
        x0 = np.random.default_rng(current_seed).uniform(-0.5 * safe_r, 0.5 * safe_r, size=problem.dimension)
        try:
            bads = BADS(safe_obj, x0, -radius * np.ones(problem.dimension), radius * np.ones(problem.dimension), -safe_r * np.ones(problem.dimension), safe_r * np.ones(problem.dimension), non_box_cons=con, options={'max_fun_evals': remaining, 'display': 'off'})
            bads.optimize()
        except Exception:
            print('Error in BADS optimization')
        if tracker.evaluations <= prev + 1:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_lshade_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int, pop_factor: int) -> Tuple[np.ndarray, float]:
    try:
        from mealpy.evolutionary_based import SHADE
        from mealpy import FloatVar
    except ImportError:
        return (np.empty((0, 2)), 0.0)
    tracker = ObjectiveTracker(problem.objective)

    def obj_w(x):
        norm = np.linalg.norm(x)
        if norm > radius:
            tracker.evaluations += 1
            return 1000000.0 + 1000.0 * (norm - radius) ** 2
        val = tracker(x)
        return float(val) if np.isfinite(val) else 1e+300
    current_seed = seed
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        problem_dict = {'bounds': FloatVar(lb=[-radius] * problem.dimension, ub=[radius] * problem.dimension), 'minmax': 'min', 'obj_func': obj_w, 'log_to': None}
        model = SHADE.L_SHADE(epoch=10000, pop_size=pop_factor * problem.dimension)
        model.solve(problem_dict, seed=current_seed, termination={'max_fe': budget - tracker.evaluations})
        if tracker.evaluations <= prev + 1:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_nomad_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        import PyNomad
    except ImportError:
        return (np.empty((0, 2)), 0.0)
    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed
    while tracker.evaluations < budget:
        prev = tracker.evaluations
        safe_r = radius / np.sqrt(problem.dimension) * 0.9
        x0 = np.random.default_rng(current_seed).uniform(-safe_r, safe_r, size=problem.dimension).tolist()

        def bb(x):
            coords = np.array([x.get_coord(i) for i in range(x.size())])
            if np.linalg.norm(coords) > radius:
                tracker.evaluations += 1
                return 0
            val = tracker(coords)
            x.setBBO(str(val).encode('UTF-8'))
            return 1
        params = ['BB_OUTPUT_TYPE OBJ', f'MAX_BB_EVAL {budget - tracker.evaluations}', 'DISPLAY_DEGREE 0', f'SEED {current_seed}', 'DIRECTION_TYPE ORTHO 2n', 'QUAD_MODEL_SEARCH NO']
        try:
            PyNomad.optimize(bb, x0, [-radius] * problem.dimension, [radius] * problem.dimension, params)
        except Exception:
            pass
        if tracker.evaluations <= prev + 1:
            break
        current_seed += 1
    return (tracker.history_array(), tracker.elapsed_time())

def run_random_search(problem: ProblemSpec, radius: float, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    rng = np.random.default_rng(seed)
    while tracker.evaluations < budget:
        u = rng.standard_normal(problem.dimension)
        tracker(u / np.linalg.norm(u) * rng.uniform(0, 1) ** (1.0 / problem.dimension) * radius)
    return (tracker.history_array(), tracker.elapsed_time())

def run_grid_search_baseline(problem: ProblemSpec, radius: float, budget: int, seed: int, step_size: float) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    rng = np.random.default_rng(seed)
    while tracker.evaluations < budget:
        u = rng.standard_normal(problem.dimension)
        p = u / np.linalg.norm(u) * rng.uniform(0, 1) ** (1.0 / problem.dimension) * radius
        tracker(np.rint(p / step_size) * step_size)
    return (tracker.history_array(), tracker.elapsed_time())

def build_optimizer_configs(settings: BenchmarkSettings) -> List[OptimizerConfig]:
    configs = []
    if settings.include_cds:
        for h, n in itertools.product(settings.cds_h_list, settings.cds_n_list):
            configs.append(OptimizerConfig(f'CDS (h={h}, N={n})', run_cds, {'step_size': h, 'num_cells': n}))
    if settings.include_cmaes:
        for s in settings.cma_sigma_list:
            configs.append(OptimizerConfig(f'CMA-ES (sigma={s})', run_cma_es, {'sigma_scale': s}))
    if settings.include_pso:
        for sw in settings.pso_swarm_list:
            configs.append(OptimizerConfig(f'PSO (swarm={sw})', run_pso, {'swarmsize': sw}))
    if settings.include_lshade:
        for f in settings.lshade_pop_factors:
            configs.append(OptimizerConfig(f'L-SHADE (pop={f}d)', run_lshade_baseline, {'pop_factor': f}))
    if settings.include_bads:
        configs.append(OptimizerConfig('BADS', run_bads_baseline, {}))
    if settings.include_nomad:
        configs.append(OptimizerConfig('NOMAD', run_nomad_baseline, {}))
    if settings.include_pdfo:
        configs.append(OptimizerConfig('Powell (PDFO)', run_pdfo_baseline, {}))
    if settings.include_neldermead:
        configs.append(OptimizerConfig('Nelder-Mead', run_neldermead_baseline, {}))
    if settings.include_de:
        configs.append(OptimizerConfig('DE (pop=15d)', run_differential_evolution, {'popsize': 15}))
    if settings.include_grid_search:
        configs.append(OptimizerConfig('Pure Grid Search (h=0.25)', run_grid_search_baseline, {'step_size': 0.25}))
    if settings.include_random:
        configs.append(OptimizerConfig('Random Search', run_random_search, {}))
    return configs

def run_full_benchmark(settings: Optional[BenchmarkSettings]=None, output_csv: Optional[str]=None) -> pd.DataFrame:
    config = settings or BenchmarkSettings()
    optimizer_configs = build_optimizer_configs(config)
    columns = ['Problem', 'Dimension', 'Optimizer', 'Seed', 'Final Loss', 'Total Evals', 'Time (s)']
    records = []
    csv_handle = None
    csv_writer = None
    if output_csv is not None:
        csv_handle = open(output_csv, 'w', newline='')
        csv_writer = csv.DictWriter(csv_handle, fieldnames=columns)
        csv_writer.writeheader()
    try:
        for dim in config.dimensions:
            problems = build_problem_suite(dim)
            for problem in problems:
                print(f'\n===== PROBLEM: {problem.name} ({dim}D) =====')
                for opt in optimizer_configs:
                    opt_records = []
                    for seed in config.seeds:
                        print(f'  -> {opt.name} | Seed {seed}')
                        hist, t = opt.runner(problem, problem.radius, config.budget_evaluations, seed, **opt.params)
                        record = {'Problem': problem.name, 'Dimension': dim, 'Optimizer': opt.name, 'Seed': seed, 'Final Loss': float(hist[-1, 1]) if hist.size > 0 else float(np.inf), 'Total Evals': int(hist[-1, 0]) if hist.size > 0 else int(config.budget_evaluations), 'Time (s)': float(t)}
                        records.append(record)
                        opt_records.append(record)
                        if csv_writer is not None:
                            csv_writer.writerow(record)
                            csv_handle.flush()
                        print(f"     done | loss={record['Final Loss']:.6e} | evals={record['Total Evals']} | time={record['Time (s)']:.2f}s")
                    mean_loss = float(np.mean([r['Final Loss'] for r in opt_records]))
                    mean_time = float(np.mean([r['Time (s)'] for r in opt_records]))
                    print(f'  => {opt.name} summary | mean_loss={mean_loss:.6e} | mean_time={mean_time:.2f}s')
    finally:
        if csv_handle is not None:
            csv_handle.close()
    return pd.DataFrame(records)

def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    df['Optimizer_Base'] = df['Optimizer'].apply(lambda x: x.split(' (')[0])
    df_best_params = df.loc[df.groupby(['Problem', 'Dimension', 'Seed', 'Optimizer_Base'])['Final Loss'].idxmin()]
    summary = df_best_params.groupby(['Problem', 'Dimension', 'Optimizer_Base']).agg(mu=('Final Loss', 'mean'), sigma=('Final Loss', 'std'), evals=('Total Evals', 'mean')).reset_index()
    print('\n--- SUMMARY TABLE (Best Config per Method) ---')
    print(summary.to_string(index=False))
    return summary
