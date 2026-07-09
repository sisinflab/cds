from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_ORIGINAL = DEFAULT_RESULTS_DIR / "directgolib_abs_layeb_original.csv"
DEFAULT_SHIFTED = DEFAULT_RESULTS_DIR / "directgolib_abs_layeb_shifted.csv"

OPTIMIZER_ORDER = [
    "CDS (Ours)",
    "CMA-ES",
    "BADS",
    "NOMAD",
    "Powell (PDFO)",
    "PSO",
    "DE",
    "L-SHADE",
    "Nelder-Mead",
    "Pure Grid Search",
    "Random Search",
]


def get_base_optimizer(name: str) -> str:
    if name.startswith("CDS") or "CDS" in name:
        return "CDS (Ours)"
    if "CMA-ES" in name:
        return "CMA-ES"
    if "PSO" in name:
        return "PSO"
    if "DE" in name and "L-SHADE" not in name:
        return "DE"
    if "L-SHADE" in name:
        return "L-SHADE"
    if "Nelder-Mead" in name:
        return "Nelder-Mead"
    if "Powell" in name:
        return "Powell (PDFO)"
    if "Grid Search" in name:
        return "Pure Grid Search"
    if "Random" in name:
        return "Random Search"
    if "BADS" in name:
        return "BADS"
    if "NOMAD" in name:
        return "NOMAD"
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bias analysis for DIRECTGOLib ABS/Layeb original vs shifted/rotated instances."
    )
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--shifted", type=Path, default=DEFAULT_SHIFTED)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument(
        "--include-layeb01",
        action="store_true",
        help="Include Layeb01 despite double-precision overflow on the default domain.",
    )
    return parser.parse_args()


def read_directgolib_csv(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    duplicated_header = df["Problem"].astype(str).eq("Problem")
    df = df.loc[~duplicated_header].copy()

    numeric_columns = [
        "Dimension",
        "Instance",
        "Rotated",
        "Seed",
        "Final Loss",
        "True Opt",
        "Opt Gap",
        "Total Evals",
        "Time (s)",
        "Shift Norm",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Dataset"] = dataset
    df["Optimizer_Base"] = df["Optimizer"].apply(get_base_optimizer)
    df["Condition"] = np.select(
        [
            df["Instance"].eq(0),
            df["Rotated"].eq(0),
            df["Rotated"].eq(1),
        ],
        ["Original", "Shift only", "Shift+rotation"],
        default="Unknown",
    )
    return df


def add_all_transformed_condition(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df[df["Condition"].isin(["Shift only", "Shift+rotation"])].copy()
    transformed["Condition"] = "All transformed"
    return pd.concat([df, transformed], ignore_index=True)


def selected_configs_from_original(original: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for (dimension, optimizer_base), group in original.groupby(["Dimension", "Optimizer_Base"]):
        finite_gap = group["Opt Gap"].replace([np.inf, -np.inf], np.nan)
        config_scores = group.assign(finite_gap=finite_gap).groupby("Optimizer")["finite_gap"].mean()
        config_scores = config_scores.fillna(np.inf)
        selected_optimizer = config_scores.idxmin()
        selected_rows.append(
            {
                "Dimension": dimension,
                "Optimizer_Base": optimizer_base,
                "Optimizer": selected_optimizer,
                "Original finite mean gap": config_scores.loc[selected_optimizer],
            }
        )
    return pd.DataFrame(selected_rows)


def apply_selected_configs(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        selected[["Dimension", "Optimizer_Base", "Optimizer"]],
        on=["Dimension", "Optimizer_Base", "Optimizer"],
        how="inner",
    )


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    grouped = [
        "Source",
        "Family",
        "Condition",
        "Dimension",
        "Optimizer_Base",
    ]
    agg = (
        df.groupby(grouped)
        .agg(
            mu_gap=("Opt Gap", "mean"),
            finite_mu_gap=("Opt Gap", lambda values: values.replace([np.inf, -np.inf], np.nan).mean()),
            finite_rate=("Opt Gap", lambda values: np.isfinite(values).mean()),
            mu_time=("Time (s)", "mean"),
            n_runs=("Opt Gap", "size"),
            n_instances=("Instance", "nunique"),
        )
        .reset_index()
    )
    agg["Rank"] = agg.groupby(["Source", "Condition", "Dimension"])["mu_gap"].rank(method="average")
    return agg


def ordered_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in OPTIMIZER_ORDER if column in table.columns]


def render_table(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown()
    except ImportError:
        return table.to_string()


def print_pivot(table: pd.DataFrame, title: str, digits: int = 2) -> None:
    cols = ordered_columns(table)
    if cols:
        table = table[cols]
    print(f"\n--- {title} ---")
    print(render_table(table.round(digits)))


def print_selected_configs(selected: pd.DataFrame) -> None:
    table = selected.sort_values(["Dimension", "Optimizer_Base"]).set_index(["Dimension", "Optimizer_Base"])
    print("\n--- SELECTED CONFIGURATIONS FROM ORIGINAL INSTANCES ---")
    print(render_table(table))


def print_average_rank_tables(agg: pd.DataFrame, dimension: int) -> None:
    df_dim = agg[agg["Dimension"].eq(dimension)].copy()
    condition_rank = df_dim.pivot_table(
        index="Condition",
        columns="Optimizer_Base",
        values="Rank",
        aggfunc="mean",
    ).reindex(["Original", "Shift only", "Shift+rotation", "All transformed"])
    print_pivot(condition_rank, f"TABLE 1: Average Rank by Transformation ({dimension}D)")

    family_rank = df_dim.pivot_table(
        index=["Family", "Condition"],
        columns="Optimizer_Base",
        values="Rank",
        aggfunc="mean",
    )
    family_rank = family_rank.reindex(
        pd.MultiIndex.from_product(
            [["ABS", "Layeb"], ["Original", "Shift only", "Shift+rotation", "All transformed"]],
            names=["Family", "Condition"],
        )
    )
    print_pivot(family_rank, f"TABLE 2: Average Rank by Family and Transformation ({dimension}D)")


def print_cds_delta_table(agg: pd.DataFrame, dimension: int) -> None:
    cds = agg[(agg["Dimension"].eq(dimension)) & (agg["Optimizer_Base"].eq("CDS (Ours)"))].copy()
    avg_rank = (
        cds.groupby(["Family", "Condition"])
        .agg(avg_rank=("Rank", "mean"), n_sources=("Source", "nunique"), finite_rate=("finite_rate", "mean"))
        .reset_index()
    )
    original = avg_rank[avg_rank["Condition"].eq("Original")][["Family", "avg_rank"]].rename(
        columns={"avg_rank": "original_avg_rank"}
    )
    avg_rank = avg_rank.merge(original, on="Family", how="left")
    avg_rank["delta_vs_original"] = avg_rank["avg_rank"] - avg_rank["original_avg_rank"]
    avg_rank = avg_rank[avg_rank["Condition"].ne("Original")]
    avg_rank = avg_rank.set_index(["Family", "Condition"]).sort_index()
    print("\n--- TABLE 3: CDS Rank Change Under Transformations ---")
    print(render_table(avg_rank.round(3)))


def print_log_gap_shift_table(agg: pd.DataFrame, dimension: int) -> None:
    cds = agg[(agg["Dimension"].eq(dimension)) & (agg["Optimizer_Base"].eq("CDS (Ours)"))].copy()
    original = cds[cds["Condition"].eq("Original")][["Source", "Family", "mu_gap"]].rename(
        columns={"mu_gap": "original_gap"}
    )
    rows = []
    for condition in ["Shift only", "Shift+rotation", "All transformed"]:
        current = cds[cds["Condition"].eq(condition)][["Source", "Family", "mu_gap"]].rename(
            columns={"mu_gap": "transformed_gap"}
        )
        merged = current.merge(original, on=["Source", "Family"], how="inner")
        finite = merged[np.isfinite(merged["original_gap"]) & np.isfinite(merged["transformed_gap"])].copy()
        finite["delta_log10_gap"] = np.log10(finite["transformed_gap"] + 1e-12) - np.log10(
            finite["original_gap"] + 1e-12
        )
        for family, group in finite.groupby("Family"):
            rows.append(
                {
                    "Family": family,
                    "Condition": condition,
                    "median_delta_log10_gap": group["delta_log10_gap"].median(),
                    "q25_delta": group["delta_log10_gap"].quantile(0.25),
                    "q75_delta": group["delta_log10_gap"].quantile(0.75),
                    "n_sources": group["Source"].nunique(),
                }
            )
    table = pd.DataFrame(rows).set_index(["Family", "Condition"]).sort_index()
    print("\n--- TABLE 4: CDS Log10 Gap Change vs Original (Finite Sources Only) ---")
    print(render_table(table.round(3)))


def print_rank_stability(agg: pd.DataFrame, dimension: int) -> None:
    df_dim = agg[agg["Dimension"].eq(dimension)].copy()
    avg_rank = (
        df_dim.groupby(["Condition", "Optimizer_Base"])["Rank"]
        .mean()
        .unstack("Optimizer_Base")
        .reindex(["Original", "Shift only", "Shift+rotation", "All transformed"])
    )
    original = avg_rank.loc["Original"]
    rows = []
    for condition in ["Shift only", "Shift+rotation", "All transformed"]:
        current = avg_rank.loc[condition]
        common = original.dropna().index.intersection(current.dropna().index)
        original_ranks = original[common].rank()
        current_ranks = current[common].rank()
        rows.append(
            {
                "Condition": condition,
                "Spearman rank corr. vs Original": original_ranks.corr(current_ranks),
            }
        )
    table = pd.DataFrame(rows).set_index("Condition")
    print("\n--- TABLE 5: Optimizer Ranking Stability ---")
    print(render_table(table.round(3)))


def print_finite_rate_table(agg: pd.DataFrame, dimension: int) -> None:
    df_dim = agg[agg["Dimension"].eq(dimension)].copy()
    finite_rate = df_dim.pivot_table(
        index="Condition",
        columns="Optimizer_Base",
        values="finite_rate",
        aggfunc="mean",
    ).reindex(["Original", "Shift only", "Shift+rotation", "All transformed"])
    print_pivot(finite_rate, f"TABLE 6: Mean Finite-Run Rate by Transformation ({dimension}D)", digits=3)


def print_pairwise_cds_table(agg: pd.DataFrame, dimension: int) -> None:
    df_dim = agg[agg["Dimension"].eq(dimension)].copy()
    rows = []
    comparators = ["Pure Grid Search", "Random Search", "CMA-ES"]
    for condition in ["Original", "Shift only", "Shift+rotation", "All transformed"]:
        condition_df = df_dim[df_dim["Condition"].eq(condition)]
        pivot = condition_df.pivot_table(
            index=["Source", "Family"],
            columns="Optimizer_Base",
            values="mu_gap",
            aggfunc="first",
        )
        if "CDS (Ours)" not in pivot.columns:
            continue
        for comparator in comparators:
            if comparator not in pivot.columns:
                continue
            pair = pivot[["CDS (Ours)", comparator]].replace([np.inf, -np.inf], np.nan).dropna()
            if pair.empty:
                continue
            log_ratio = np.log10(pair["CDS (Ours)"] + 1e-12) - np.log10(pair[comparator] + 1e-12)
            rows.append(
                {
                    "Condition": condition,
                    "Comparator": comparator,
                    "CDS win rate": (pair["CDS (Ours)"] < pair[comparator]).mean(),
                    "median log10(CDS/comparator)": log_ratio.median(),
                    "n_sources": len(pair),
                }
            )
    table = pd.DataFrame(rows).set_index(["Condition", "Comparator"])
    print("\n--- TABLE 7: Pairwise CDS Robustness Against Reference Optimizers ---")
    print(render_table(table.round(3)))


def print_data_audit(raw: pd.DataFrame, filtered: pd.DataFrame, excluded_sources: list[str]) -> None:
    print("DIRECTGOLib center-bias analysis")
    print(f"Raw rows: {len(raw):,} | Filtered rows: {len(filtered):,}")
    print(
        f"Sources used: {filtered['Source'].nunique()} "
        f"({', '.join(sorted(filtered['Family'].unique()))})"
    )
    print(f"Instances used: {sorted(filtered['Instance'].dropna().astype(int).unique().tolist())}")
    if excluded_sources:
        print(f"Excluded sources: {', '.join(excluded_sources)}")
    print(
        "Protocol: select one hyperparameter configuration per optimizer on original instances, "
        "then reuse it unchanged on shifted/rotated instances."
    )


def main() -> None:
    args = parse_args()
    original = read_directgolib_csv(args.original, "original")
    shifted = read_directgolib_csv(args.shifted, "shifted")
    raw = pd.concat([original, shifted], ignore_index=True)
    raw = add_all_transformed_condition(raw)
    raw = raw[raw["Dimension"].eq(args.dimension)].copy()

    excluded_sources = []
    if not args.include_layeb01:
        excluded_sources.append("Layeb01")

    filtered = raw[~raw["Source"].isin(excluded_sources)].copy()
    original_filtered = filtered[filtered["Condition"].eq("Original")].copy()
    selected = selected_configs_from_original(original_filtered)
    selected_runs = apply_selected_configs(filtered, selected)
    agg = aggregate_runs(selected_runs)

    print_data_audit(raw, selected_runs, excluded_sources)
    print_selected_configs(selected)
    print_average_rank_tables(agg, args.dimension)
    print_cds_delta_table(agg, args.dimension)
    print_log_gap_shift_table(agg, args.dimension)
    print_rank_stability(agg, args.dimension)
    print_finite_rate_table(agg, args.dimension)
    print_pairwise_cds_table(agg, args.dimension)


if __name__ == "__main__":
    main()
