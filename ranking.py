import pandas as pd
import numpy as np
import ioh
import re
from pathlib import Path


def get_true_optimum(problem_name, dim):
    """Return the theoretical global optimum (f_opt)."""
    if "BBOB" in problem_name:
        match = re.search(r'F(\d+)', problem_name)
        if match:
            fid = int(match.group(1))
            p = ioh.get_problem(fid, instance=1, dimension=dim, problem_class=ioh.ProblemClass.BBOB)
            return p.optimum.y
    if "ABS" in problem_name:
        return 0.0
    if "Linear Regression" in problem_name:
        return None
    if "Layeb" in problem_name:
        return 0.0

    return None


def analyze_with_true_optima(csv_path):
    df = pd.read_csv(csv_path, sep=';')

    df['Final Loss (mu)'] = df['Final Loss (mu)'].astype(float)
    df['Final Loss (sigma)'] = df['Final Loss (sigma)'].astype(float)
    df['Mean_Evals'] = df['Mean_Evals'].astype(float)
    def get_base_optimizer(name):
        if name.startswith("CDS"): return "CDS (Ours)"
        if "CMA-ES" in name: return "CMA-ES"
        if "PSO" in name: return "PSO"
        if "DE" in name: return "DE"
        if "L-SHADE" in name: return "L-SHADE"
        if "Nelder-Mead" in name: return "Nelder-Mead"
        if "Powell" in name: return "Powell"
        if "Grid" in name: return "Pure Grid Search"
        if "Random" in name: return "Random Search"
        if "BADS" in name: return "BADS"
        if "NOMAD" in name: return "NOMAD"
        if "TuRBO" in name: return "TuRBO"
        return name

    df['Optimizer_Base'] = df['Optimizer'].apply(get_base_optimizer)
    results_with_gap = []

    for (prob, dim), group in df.groupby(['Problem', 'Dimension']):
        f_opt = get_true_optimum(prob, dim)
        if f_opt is None:
            f_opt = group['Final Loss (mu)'].min()

        group = group.copy()
        group['Optimality_Gap'] = (group['Final Loss (mu)'] - f_opt).clip(lower=0)
        results_with_gap.append(group)

    df_gap = pd.concat(results_with_gap)
    df_best = df_gap.loc[df_gap.groupby(['Problem', 'Dimension', 'Optimizer_Base'])['Optimality_Gap'].idxmin()]
    df_best['Rank'] = df_best.groupby(['Problem', 'Dimension'])['Optimality_Gap'].rank(method='min')
    for dim in df_best['Dimension'].unique():
        print(f"\n" + "=" * 60)
        print(f" AVERAGE RANK TABLE - DIMENSION {dim}D")
        print("=" * 60)

        dim_df = df_best[df_best['Dimension'] == dim]
        pivot_rank = dim_df.pivot_table(
            index='Problem',
            columns='Optimizer_Base',
            values='Rank',
            aggfunc='mean'
        ).round(2)
        cols = ['CDS (Ours)'] + [c for c in pivot_rank.columns if c != 'CDS (Ours)']
        pivot_rank = pivot_rank[cols]

        print(pivot_rank.to_string())
        print(f"\n[Tactical analysis {dim}D]: Functions where CDS is Top 1 (Rank 1.0):")
        top_functions = dim_df[(dim_df['Optimizer_Base'] == 'CDS (Ours)') & (dim_df['Rank'] == 1.0)]['Problem'].unique()
        for p in top_functions:
            val = \
            dim_df[(dim_df['Problem'] == p) & (dim_df['Optimizer_Base'] == 'CDS (Ours)')]['Final Loss (mu)'].values[0]
            print(f" - {p}: Loss {val:.4e}")
    df_best.to_csv("results/results_per_paper_best_only.csv", index=False)
    return df_best



if __name__ == "__main__":
    candidates = sorted(Path("results").glob("*no_bads*.tsv.csv"))
    path = str(candidates[0]) if candidates else "results/benchmark_no_bads.tsv.csv"
    try:
        analyze_with_true_optima(path)
    except FileNotFoundError:
        print(f"File {path} not found. Make sure benchmark data has been generated.")
