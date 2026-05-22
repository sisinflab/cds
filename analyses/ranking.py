import pandas as pd
import numpy as np
import re
import ioh
FILE_PATH = '../results/cds_benchmarking_complete_results.csv'

def get_base_optimizer(name):
    if 'CDS' in name:
        return 'CDS (Ours)'
    if 'CMA-ES' in name:
        return 'CMA-ES'
    if 'PSO' in name:
        return 'PSO'
    if 'DE' in name and 'L-SHADE' not in name:
        return 'DE'
    if 'L-SHADE' in name:
        return 'L-SHADE'
    if 'Nelder-Mead' in name:
        return 'Nelder-Mead'
    if 'Powell' in name:
        return 'Powell (PDFO)'
    if 'Grid Search' in name:
        return 'Pure Grid Search'
    if 'Random' in name:
        return 'Random Search'
    if 'BADS' in name:
        return 'BADS'
    if 'NOMAD' in name:
        return 'NOMAD'
    return name

def get_problem_class(prob_name):
    if 'Linear Regression' in prob_name:
        return '6. Convex (Sanity Check)'
    if 'ABS' in prob_name or 'Layeb' in prob_name:
        return '5. Irregular / Non-Smooth'
    match = re.search('F(\\d+)', prob_name)
    if not match:
        return 'Unknown'
    fid = int(match.group(1))
    if 1 <= fid <= 5:
        return '1. Separable'
    if 6 <= fid <= 9:
        return '2. Moderate Cond.'
    if 10 <= fid <= 14:
        return '3. High Cond.'
    if 15 <= fid <= 19:
        return '4. Multi-modal (Global)'
    if 20 <= fid <= 24:
        return '5. Multi-modal (Weak)'
    return 'Unknown'

def get_true_optimum(prob_name, dim):
    clean_name = re.sub('\\s*\\(?[\\d]+D\\)?\\s*', '', prob_name).strip()
    if 'BBOB' in clean_name:
        match = re.search('F(\\d+)', clean_name)
        if match:
            fid = int(match.group(1))
            try:
                p = ioh.get_problem(fid, instance=1, dimension=dim, problem_class=ioh.ProblemClass.BBOB)
                return p.optimum.y
            except:
                pass
    if 'ABS' in clean_name or 'Layeb' in clean_name:
        return 0.0
    return None
print('Data Analysis started (Anti-Bias Protocol)...')
df = pd.read_csv(FILE_PATH, sep=';')
df['Problem_Clean'] = df['Problem'].str.replace('\\s*\\(?[\\d]+D\\)?\\s*', '', regex=True).str.strip()
df['Optimizer_Base'] = df['Optimizer'].apply(get_base_optimizer)
df['Class'] = df['Problem_Clean'].apply(get_problem_class)
df['f_opt'] = df.apply(lambda row: get_true_optimum(row['Problem_Clean'], row['Dimension']), axis=1)
min_found = df.groupby(['Problem_Clean', 'Dimension', 'Seed'])['Final Loss'].transform('min')
df['f_opt'] = df['f_opt'].fillna(min_found)
df['Opt_Gap'] = (df['Final Loss'] - df['f_opt']).clip(lower=0)
df_configs = df.groupby(['Problem_Clean', 'Dimension', 'Optimizer_Base', 'Optimizer'])['Opt_Gap'].mean().reset_index()
idx_best_configs = df_configs.groupby(['Problem_Clean', 'Dimension', 'Optimizer_Base'])['Opt_Gap'].idxmin()
best_configs_list = df_configs.loc[idx_best_configs, ['Problem_Clean', 'Dimension', 'Optimizer']]
df_final = pd.merge(df, best_configs_list, on=['Problem_Clean', 'Dimension', 'Optimizer'])
df_agg = df_final.groupby(['Problem_Clean', 'Class', 'Dimension', 'Optimizer_Base']).agg(mu_gap=('Opt_Gap', 'mean'), std_gap=('Opt_Gap', 'std'), mu_time=('Time (s)', 'mean')).reset_index()
df_agg['Rank'] = df_agg.groupby(['Problem_Clean', 'Dimension'])['mu_gap'].rank(method='min')

def print_latex_pivot(pivot_df, title):
    col_order = ['CDS (Ours)', 'CMA-ES', 'BADS', 'NOMAD', 'Powell (PDFO)', 'PSO', 'DE', 'L-SHADE', 'Nelder-Mead', 'Pure Grid Search', 'Random Search']
    cols = [c for c in col_order if c in pivot_df.columns]
    print(f'\n--- {title} ---')
    print(pivot_df[cols].round(2).to_markdown())
tab1 = df_agg[df_agg['Dimension'] == 10].pivot_table(index='Class', columns='Optimizer_Base', values='Rank', aggfunc='mean')
print_latex_pivot(tab1, 'TABLE 1: Average Rank 10D')
tab2 = df_agg[df_agg['Optimizer_Base'] != 'BADS'].pivot_table(index='Dimension', columns='Optimizer_Base', values='Rank', aggfunc='mean')
print_latex_pivot(tab2, 'TABLE 2: Scalability (Global Rank)')
target_probs = ['Linear Regression', 'Rosenbrock', 'Rastrigin', 'ABS', 'Layeb']
df_deep = df_agg[(df_agg['Dimension'] == 10) & df_agg['Problem_Clean'].str.contains('|'.join(target_probs))]

def fmt(mu, std):
    return f'{mu:.2e} ± {std:.2e}' if mu < 0.01 and mu > 0 else f'{mu:.2f} ± {std:.2f}'
df_deep['Formatted'] = df_deep.apply(lambda r: fmt(r['mu_gap'], r['std_gap']), axis=1)
tab3 = df_deep.pivot(index='Problem_Clean', columns='Optimizer_Base', values='Formatted')
print_latex_pivot(tab3, 'TABLE 3: Deep-Dive Real Values (Gap 10D)')
print('\n--- TIKZ SENSITIVITY CODE (Rastrigin 10D) ---')
df_sens = df[(df['Optimizer_Base'] == 'CDS (Ours)') & df['Problem_Clean'].str.contains('Rastrigin') & (df['Dimension'] == 10)]
df_sens_agg = df_sens.groupby('Optimizer').agg(mu_gap=('Opt_Gap', 'mean')).reset_index()

def extract_h_n(name):
    h = float(re.search('h=([0-9.]+)', name).group(1))
    n = int(re.search('N=([0-9]+)', name).group(1))
    return (h, n)
df_sens_agg[['h', 'N']] = df_sens_agg['Optimizer'].apply(lambda x: pd.Series(extract_h_n(x)))
for n, group in df_sens_agg.groupby('N'):
    group = group.sort_values('h', ascending=False)
    coords = ' '.join([f"({row['h']}, {max(row['mu_gap'], 0.0001):.4f})" for _, row in group.iterrows()])
    print(f'% N={n}\n\\addplot coordinates {{ {coords} }};\n\\addlegendentry{{$N_\\mathrm{{init}}={n}$}}')
