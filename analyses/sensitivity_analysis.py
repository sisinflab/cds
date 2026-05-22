import numpy as np
import ioh
from benchmarking_module import build_bbob_problem, run_cds

def run_sensitivity():
    dim = 10
    budget = 5000
    seeds = list(range(20))
    N_values = [1, 2, 4, 8, 16, 32]
    h_values = [1.0, 0.5, 0.25, 0.125, 0.0625]
    target_problems = [15, 24]
    for fid in target_problems:
        problem = build_bbob_problem(func_id=fid, dim=dim)
        p_ioh = ioh.get_problem(fid, instance=1, dimension=dim, problem_class=ioh.ProblemClass.BBOB)
        f_opt = p_ioh.optimum.y
        print('=====================================================')
        print(f' Starting Sensitivity Analysis: {problem.name}')
        print(' (This operation should complete quickly...)')
        print('=====================================================\n')
        tikz_data = {n: [] for n in N_values}
        for n in N_values:
            print(f'--- Computing for N = {n} ---')
            for h in h_values:
                gaps = []
                for seed in seeds:
                    hist, _ = run_cds(problem=problem, radius=problem.radius, budget=budget, seed=seed, step_size=h, num_cells=n)
                    final_loss = hist[-1, 1] if hist.size > 0 else np.inf
                    gap = max(0.0, final_loss - f_opt)
                    gaps.append(gap)
                mean_gap = np.mean(gaps)
                plot_gap = 0.0001 if mean_gap < 0.0001 else mean_gap
                tikz_data[n].append(f'({h}, {plot_gap:.4f})')
                print(f'  h={h:<6} -> Mean Gap: {mean_gap:.4f}')
            print('')
        print('\n' + '=' * 60)
        print(' TIKZ CODE READY TO PASTE INTO YOUR .TEX FILE')
        print('=' * 60)
        for n in N_values:
            coords = ' '.join(tikz_data[n])
            print(f'% N_init={n}')
            print(f'\\addplot coordinates {{')
            print(f'    {coords}')
            print(f'}};')
            print(f'\\addlegendentry{{$N_\\mathrm{{init}}={n}$}}\n')
if __name__ == '__main__':
    run_sensitivity()
