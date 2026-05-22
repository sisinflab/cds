import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from benchmarking_module import build_layeb_problem, build_abs_problem, build_linear_regression_problem, build_bbob_problem, run_cds, run_cma_es, run_pso, run_differential_evolution, run_neldermead_baseline, run_pdfo_baseline, run_grid_search_baseline, run_nomad_baseline, run_random_search, run_lshade_baseline, run_bads_baseline

def get_all_runners(prob_title):
    cds_h = 0.25
    if 'Linear' in prob_title:
        cds_h = 0.125
    elif 'Layeb' in prob_title:
        cds_h = 0.0625
    return {'BADS': (run_bads_baseline, {})}
styles = {'CDS (Ours)': 'blue', 'CMA-ES': 'green', 'NOMAD': 'purple', 'Powell (PDFO)': 'orange', 'DE': 'magenta', 'PSO': 'cyan', 'Pure Grid': 'gray', 'Random': 'black', 'Nelder-Mead': 'brown', 'L-SHADE': 'olive'}
problems = [('Linear Regression', build_linear_regression_problem(dim=10)), ('BBOB F8 (Rosenbrock)', build_bbob_problem(func_id=8, dim=10)), ('BBOB F15 (Rastrigin)', build_bbob_problem(func_id=15, dim=10)), ('ABS (Non-smooth)', build_abs_problem(dim=10)), ('Layeb 1 (Irregular)', build_layeb_problem(dim=10))]
budget = 5000
seed_to_plot = 42
print('\n' + '=' * 60)
print(' GENERATING TIKZ CODE FOR CONVERGENCE CURVES (SEED 42)')
print('=' * 60)
for title, prob in problems:
    print(f'\n% --- PLOT FOR: {title} ---')
    print('\\begin{tikzpicture}')
    print(f'  \\begin{{loglogaxis}}[width=0.48\\textwidth, height=5cm, xlabel={{Function Evaluations}}, ylabel={{Loss}}, title={{{title}}}, grid=major, legend pos=outer north east, legend style={{font=\\tiny}}]')
    runners = get_all_runners(title)
    for algo_name, (runner, params) in runners.items():
        hist, _ = runner(prob, radius=prob.radius, budget=budget, seed=seed_to_plot, **params)
        if hist is not None and hist.size > 0:
            idx = np.round(np.linspace(0, len(hist) - 1, 40)).astype(int)
            sampled_hist = hist[idx]
            coords = ' '.join([f'({x:.0f}, {max(y, 1e-08):.4e})' for x, y in sampled_hist if x > 0])
            color = styles.get(algo_name, 'black')
            thick_style = 'ultra thick' if algo_name == 'CDS (Ours)' else 'thick'
            if algo_name in ['Pure Grid', 'Random']:
                thick_style += ', dashed'
            print(f'    \\addplot[color={color}, {thick_style}] coordinates {{ {coords} }};')
            print(f'    \\addlegendentry{{{algo_name}}}')
    print('  \\end{loglogaxis}')
    print('\\end{tikzpicture}')
