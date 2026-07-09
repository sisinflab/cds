from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.directgolib_ranking import (
    OPTIMIZER_ORDER,
    read_complete_results,
    render_table,
    select_fair_best_configs,
)


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "results" / "cds_benchmarking_complete_results.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ranking analysis for BBOB and linear-regression benchmarks.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument(
        "--deep-dive",
        nargs="+",
        default=["Linear Regression", "F8", "F15", "F19", "F22"],
        help="Problem-name fragments included in the detailed gap table.",
    )
    parser.add_argument(
        "--sensitivity-fragment",
        default="F15",
        help="Problem-name fragment used for CDS hyperparameter sensitivity coordinates.",
    )
    return parser.parse_args(argv)


def get_problem_class(source: str) -> str:
    if "Linear Regression" in source:
        return "6. Convex sanity check"

    match = re.search(r"F(\d+)", source)
    if not match:
        return "Unknown"

    fid = int(match.group(1))
    if 1 <= fid <= 5:
        return "1. Separable"
    if 6 <= fid <= 9:
        return "2. Moderate conditioning"
    if 10 <= fid <= 14:
        return "3. High conditioning"
    if 15 <= fid <= 19:
        return "4. Multimodal global structure"
    if 20 <= fid <= 24:
        return "5. Multimodal weak structure"
    return "Unknown"


def ordered_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in OPTIMIZER_ORDER if column in table.columns]


def print_pivot(table: pd.DataFrame, title: str, digits: int = 2) -> None:
    cols = ordered_columns(table)
    if cols:
        table = table[cols]
    print(f"\n--- {title} ---")
    print(render_table(table.round(digits)))


def build_aggregate(df_best: pd.DataFrame) -> pd.DataFrame:
    df_best = df_best.copy()
    df_best["Class"] = df_best["Source"].apply(get_problem_class)
    agg = (
        df_best.groupby(["Source", "Class", "Dimension", "Optimizer_Base"])
        .agg(
            mu_gap=("Opt Gap", "mean"),
            std_gap=("Opt Gap", "std"),
            mu_time=("Time (s)", "mean"),
            mu_evals=("Total Evals", "mean"),
            n_runs=("Opt Gap", "size"),
        )
        .reset_index()
    )
    agg["Rank"] = agg.groupby(["Source", "Dimension"])["mu_gap"].rank(method="min")
    return agg


def format_gap(mu: float, std: float) -> str:
    if pd.isna(mu):
        return "nan"
    if np.isinf(mu):
        return "inf"
    if pd.isna(std):
        std = 0.0
    if 0 < abs(mu) < 0.01 or abs(mu) >= 1e4 or abs(std) >= 1e4:
        return f"{mu:.2e} +/- {std:.2e}"
    return f"{mu:.2f} +/- {std:.2f}"


def print_average_rank_by_class(df_dim: pd.DataFrame, dimension: int) -> None:
    table = df_dim.pivot_table(
        index="Class",
        columns="Optimizer_Base",
        values="Rank",
        aggfunc="mean",
    )
    table.loc[f"OVERALL {dimension}D"] = df_dim.groupby("Optimizer_Base")["Rank"].mean()
    print_pivot(table, f"TABLE 1: Average Rank {dimension}D by Problem Class")


def print_scalability(df_agg: pd.DataFrame) -> None:
    table = df_agg.pivot_table(
        index="Dimension",
        columns="Optimizer_Base",
        values="Rank",
        aggfunc="mean",
    )
    print_pivot(table, "TABLE 2: Average Rank by Dimension")


def print_grid_ablation(df_dim: pd.DataFrame) -> None:
    ablation = df_dim[df_dim["Optimizer_Base"].isin(["CDS (Ours)", "Pure Grid Search"])]
    table = ablation.pivot_table(
        index="Source",
        columns="Optimizer_Base",
        values="mu_gap",
        aggfunc="first",
    )
    if {"CDS (Ours)", "Pure Grid Search"} - set(table.columns):
        return

    table = table[["CDS (Ours)", "Pure Grid Search"]].copy()
    table["Grid/CDS"] = table["Pure Grid Search"] / table["CDS (Ours)"].replace(0, np.nan)
    table["CDS wins"] = table["Grid/CDS"] > 1
    print("\n--- TABLE 3: CDS vs Pure Grid Search ---")
    print(render_table(table.round(4)))


def print_deep_dive(df_dim: pd.DataFrame, fragments: list[str]) -> None:
    mask = df_dim["Source"].apply(lambda source: any(fragment in source for fragment in fragments))
    deep = df_dim[mask].copy()
    if deep.empty:
        return

    deep["Gap"] = deep.apply(lambda row: format_gap(row["mu_gap"], row["std_gap"]), axis=1)
    table = deep.pivot_table(
        index="Source",
        columns="Optimizer_Base",
        values="Gap",
        aggfunc="first",
    )
    cols = ordered_columns(table)
    if cols:
        table = table[cols]
    print("\n--- TABLE 4: Detailed Optimality Gaps ---")
    print(render_table(table))


def print_time_table(df_dim: pd.DataFrame, dimension: int) -> None:
    table = (
        df_dim.groupby("Optimizer_Base")
        .agg(mean_time_s=("mu_time", "mean"), mean_evals=("mu_evals", "mean"))
        .sort_values("mean_time_s")
    )
    print(f"\n--- TABLE 5: Mean Runtime in {dimension}D ---")
    print(render_table(table.round(2)))


def extract_cds_h_n(name: str) -> tuple[float | None, int | None]:
    h_match = re.search(r"h=([0-9.]+)", name)
    n_match = re.search(r"(?:cells|N)=([0-9]+)", name)
    h_value = float(h_match.group(1)) if h_match else None
    n_value = int(n_match.group(1)) if n_match else None
    return h_value, n_value


def print_sensitivity_coordinates(raw: pd.DataFrame, dimension: int, fragment: str) -> None:
    mask = (
        raw["Optimizer_Base"].eq("CDS (Ours)")
        & raw["Source"].str.contains(fragment, na=False)
        & raw["Dimension"].eq(dimension)
    )
    cds = raw[mask].copy()
    if cds.empty:
        return

    sensitivity = cds.groupby("Optimizer").agg(mu_gap=("Opt Gap", "mean")).reset_index()
    sensitivity[["h", "N"]] = sensitivity["Optimizer"].apply(lambda name: pd.Series(extract_cds_h_n(name)))
    sensitivity = sensitivity.dropna(subset=["h", "N"])
    if sensitivity.empty:
        return

    print(f"\n--- TIKZ: CDS Sensitivity Coordinates ({fragment}, {dimension}D) ---")
    for n_value, group in sensitivity.groupby("N"):
        group = group.sort_values("h", ascending=False)
        coords = " ".join(
            f"({row.h}, {max(row.mu_gap, 0.0001):.4f})" for row in group.itertuples(index=False)
        )
        print(f"% N_init={int(n_value)}")
        print(rf"\addplot coordinates {{ {coords} }};")
        print(rf"\addlegendentry{{$N_\mathrm{{init}}={int(n_value)}$}}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df, excluded_count, warning = read_complete_results(args.csv)
    df_best, selected = select_fair_best_configs(df)
    df_agg = build_aggregate(df_best)

    available_dimensions = sorted(df_agg["Dimension"].dropna().astype(int).unique().tolist())
    dimension = args.dimension if args.dimension in available_dimensions else available_dimensions[0]
    df_dim = df_agg[df_agg["Dimension"].eq(dimension)].copy()

    print("BBOB and linear-regression ranking analysis")
    print(f"Input CSV: {args.csv}")
    print(f"Rows: {len(df):,} | Sources: {df['Source'].nunique()} | Dimensions: {available_dimensions}")
    if excluded_count:
        print(f"Excluded DIRECTGOLib-like rows from complete CSV: {excluded_count:,}")
    if warning:
        print(f"Warning: {warning}")
    print("Selection rule: one best hyperparameter configuration per (Dimension, Optimizer_Base).")

    selected_table = selected.sort_values(["Dimension", "Optimizer_Base"]).set_index(
        ["Dimension", "Optimizer_Base"]
    )
    print("\n--- SELECTED CONFIGURATIONS ---")
    print(render_table(selected_table))

    print_average_rank_by_class(df_dim, dimension)
    print_scalability(df_agg)
    print_grid_ablation(df_dim)
    print_deep_dive(df_dim, args.deep_dive)
    print_time_table(df_dim, dimension)
    print_sensitivity_coordinates(df, dimension, args.sensitivity_fragment)


if __name__ == "__main__":
    main()
