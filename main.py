import argparse
import pandas as pd
from benchmarking_module import BenchmarkSettings, create_summary_table, run_full_benchmark
from hpo_module import main as hpo_main, run_quick_benchmark as run_hpo_quick_benchmark

def quick_run() -> pd.DataFrame:
    """Quick run to verify all optimizers and dependencies execute without errors."""
    print('=====================================================')
    print(' STARTING QUICK RUN (Integrity Check)')
    print(' Budget: 1000 | Seeds: 1 | Dimensions: 2D')
    print('=====================================================\n')
    settings = BenchmarkSettings(budget_evaluations=1000, seeds=(42,), dimensions=(2,), include_cds=True, include_cmaes=True, include_pso=True, include_de=True, include_random=True, include_neldermead=True, include_pdfo=True, include_grid_search=True, include_bads=True, include_nomad=True, include_lshade=True)
    results_df = run_full_benchmark(settings)
    create_summary_table(results_df)
    return results_df

def main_cli():
    parser = argparse.ArgumentParser(description='Cellular Direct Search (CDS) - Benchmarking Suite', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', nargs='?', default='benchmark', choices=['benchmark', 'quick', 'hpo', 'hpo-quick', 'directgolib'], help='Script execution mode')
    parser.add_argument('--budget', type=int, default=5000, help='Total objective-evaluation budget')
    parser.add_argument('--seeds', type=int, default=20, help='Number of independent restarts (different seeds)')
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 20, 30, 40, 50], help='Problem dimensions to test')
    parser.add_argument('--out', type=str, default='benchmark_results.csv', help='Output CSV filename')
    parser.add_argument('--start-seed', type=int, nargs='?')
    parser.add_argument('--end-seed', type=int, nargs='?')
    parser.add_argument('--instances', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='DIRECTGOLib shifted/rotated instances to test')
    parser.add_argument('--families', type=str, nargs='+', default=['ABS', 'Layeb'], choices=['ABS', 'Layeb', 'abs', 'layeb'], help='DIRECTGOLib families to include')
    group = parser.add_argument_group('Model Exclusion (Use flags below to skip selected optimizers)')
    group.add_argument('--skip-cds', action='store_true', help='Disable Cellular Direct Search')
    group.add_argument('--skip-cmaes', action='store_true', help='Disable CMA-ES')
    group.add_argument('--skip-pso', action='store_true', help='Disable PSO')
    group.add_argument('--skip-de', action='store_true', help='Disable Differential Evolution')
    group.add_argument('--skip-random', action='store_true', help='Disable Random Search')
    group.add_argument('--skip-neldermead', action='store_true', help='Disable Nelder-Mead (SciPy)')
    group.add_argument('--skip-pdfo', action='store_true', help='Disable Powell (PDFO)')
    group.add_argument('--skip-grid', action='store_true', help='Disable Pure Grid Search')
    group.add_argument('--skip-bads', action='store_true', help='Disable BADS (recommended for faster runs)')
    group.add_argument('--skip-nomad', action='store_true', help='Disable NOMAD')
    group.add_argument('--skip-lshade', action='store_true', help='Disable L-SHADE')
    args = parser.parse_args()
    if args.mode == 'benchmark':
        print('=====================================================')
        print(' STARTING OFFICIAL BENCHMARK RUN (IEEE Access Revision)')
        print('=====================================================')
        print(f' Tested dimensions  : {args.dims}')
        print(f' Eval budget        : {args.budget}')
        if (args.start_seed is None) != (args.end_seed is None):
            parser.error('Specify both --start-seed and --end-seed, or neither.')
        if args.start_seed is not None:
            if args.end_seed <= args.start_seed:
                parser.error('--end-seed must be greater than --start-seed.')
            seeds = tuple(range(args.start_seed, args.end_seed))
        else:
            if args.seeds <= 0:
                parser.error('--seeds must be > 0.')
            seeds = tuple(range(args.seeds))
        print(f' Seed range         : ({seeds[0]} to {seeds[-1]})')
        print(f' Output file        : {args.out}')
        print('=====================================================\n')
        settings_kwargs = {'budget_evaluations': args.budget, 'seeds': seeds, 'dimensions': tuple(args.dims), 'include_cds': not args.skip_cds, 'include_cmaes': not args.skip_cmaes, 'include_pso': not args.skip_pso, 'include_de': not args.skip_de, 'include_random': not args.skip_random, 'include_neldermead': not args.skip_neldermead, 'include_pdfo': not args.skip_pdfo, 'include_grid_search': not args.skip_grid, 'include_bads': not args.skip_bads, 'include_nomad': not args.skip_nomad, 'include_lshade': not args.skip_lshade}
        settings = BenchmarkSettings(**settings_kwargs)
        results_df = run_full_benchmark(settings, output_csv=args.out)
        print(f"\n[+] RUN COMPLETED! Results were also streamed to '{args.out}' during execution.")
        create_summary_table(results_df)
    elif args.mode == 'quick':
        quick_run()
    elif args.mode == 'hpo':
        hpo_main()
    elif args.mode == 'hpo-quick':
        run_hpo_quick_benchmark()
    elif args.mode == 'directgolib':
        from directgolib_abs_layeb import DirectGOLibBenchmarkSettings, run_directgolib_benchmark

        if (args.start_seed is None) != (args.end_seed is None):
            parser.error('Specify both --start-seed and --end-seed, or neither.')
        if args.start_seed is not None:
            if args.end_seed <= args.start_seed:
                parser.error('--end-seed must be greater than --start-seed.')
            seeds = tuple(range(args.start_seed, args.end_seed))
        else:
            if args.seeds <= 0:
                parser.error('--seeds must be > 0.')
            seeds = tuple(range(args.seeds))
        instances = tuple(args.instances)
        if any(instance < 1 or instance > 5 for instance in instances):
            parser.error('--instances must contain only DIRECTGOLib instances 1..5.')
        families = tuple('ABS' if family.lower() == 'abs' else 'Layeb' for family in args.families)
        print('=====================================================')
        print(' STARTING DIRECTGOLib ABS/Layeb SHIFTED BENCHMARK')
        print('=====================================================')
        print(f' Families           : {families}')
        print(f' Instances          : {instances} (1-2 shifted, 3-5 shifted+rotated)')
        print(f' Tested dimensions  : {args.dims}')
        print(f' Eval budget        : {args.budget}')
        print(f' Seed range         : ({seeds[0]} to {seeds[-1]})')
        print(f' Output file        : {args.out}')
        print('=====================================================\n')
        settings = DirectGOLibBenchmarkSettings(
            budget_evaluations=args.budget,
            seeds=seeds,
            dimensions=tuple(args.dims),
            instances=instances,
            families=families,
            include_cds=not args.skip_cds,
            include_cmaes=not args.skip_cmaes,
            include_pso=not args.skip_pso,
            include_de=not args.skip_de,
            include_random=not args.skip_random,
            include_neldermead=not args.skip_neldermead,
            include_pdfo=not args.skip_pdfo,
            include_grid_search=not args.skip_grid,
            include_bads=not args.skip_bads,
            include_nomad=not args.skip_nomad,
            include_lshade=not args.skip_lshade,
        )
        run_directgolib_benchmark(settings, output_csv=args.out)
        print(f"\n[+] DIRECTGOLib run completed! Results were streamed to '{args.out}'.")
if __name__ == '__main__':
    main_cli()
