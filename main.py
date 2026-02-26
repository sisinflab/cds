import sys

from benchmarking_module import (
    BenchmarkSettings,
    OptimizerConfig,
    ProblemSpec,
    build_linear_regression_problem,
    build_optimizer_configs,
    build_problem_suite,
    build_rastrigin_problem,
    build_rosenbrock_problem,
    build_shifted_rotated_rastrigin_problem,
    create_summary_table,
    main,
    plot_pareto_hyperparameters,
    plot_results,
    run_cache_exploration,
    run_cds,
    run_cma_es,
    run_differential_evolution,
    run_full_benchmark,
    run_pgd,
    run_pso,
    run_quick_benchmark,
    run_radius_sensitivity_ablation,
    run_random_search,
    run_scipy_baseline,
    run_sensitivity_analysis,
)
from hpo_module import HPOSettings, main as hpo_main, run_quick_benchmark as run_hpo_quick_benchmark


def quick_run():
    return run_quick_benchmark()


def quick_hpo_run():
    return run_hpo_quick_benchmark()


__all__ = [
    "BenchmarkSettings",
    "HPOSettings",
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
    "quick_run",
    "run_radius_sensitivity_ablation",
    "run_random_search",
    "run_scipy_baseline",
    "run_sensitivity_analysis",
    "quick_hpo_run",
]


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "benchmark"
    if mode == "benchmark":
        main()
    elif mode == "quick":
        quick_run()
    elif mode == "hpo":
        hpo_main()
    elif mode == "hpo-quick":
        quick_hpo_run()
    else:
        print("Usage: python main.py [benchmark|quick|hpo|hpo-quick]")
        raise SystemExit(2)
