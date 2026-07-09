from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ioh
except ImportError:
    ioh = None


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CSV = DEFAULT_RESULTS_DIR / "directgolib_abs_layeb_original.csv"
DEFAULT_LEGACY_CSV = DEFAULT_RESULTS_DIR / "cds_benchmarking_complete_results.csv"

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


def get_optimizer_config_key(name: str) -> str:
    config_name = str(name)
    if config_name.startswith("CDS-Box"):
        return config_name.replace("CDS-Box", "CDS", 1)
    return config_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ranking analysis for DIRECTGOLib ABS/Layeb benchmark results."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV path. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=10,
        help="Dimension used for the 10D-style tables. Default: 10.",
    )
    parser.add_argument(
        "--legacy-csv",
        type=Path,
        default=None,
        help=(
            "Optional legacy benchmark CSV to include in the combined wall-clock time table. "
            f"Typical value: {DEFAULT_LEGACY_CSV}"
        ),
    )
    return parser.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=";")

    df.columns = [col.strip() for col in df.columns]
    if "Problem" in df.columns:
        df = df.loc[~df["Problem"].astype(str).eq("Problem")].copy()
    required = {
        "Problem",
        "Optimizer",
        "Seed",
        "Final Loss",
        "Total Evals",
        "Time (s)",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for column in [
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
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Dimension" not in df.columns:
        raise ValueError("DIRECTGOLib ranking needs a Dimension column.")
    if "Source" not in df.columns:
        df["Source"] = df["Problem"].astype(str)
    if "Family" not in df.columns:
        df["Family"] = np.select(
            [
                df["Source"].astype(str).str.contains("ABS", case=False, na=False),
                df["Source"].astype(str).str.contains("Layeb", case=False, na=False),
            ],
            ["ABS", "Layeb"],
            default="Unknown",
        )
    if "Instance" not in df.columns:
        df["Instance"] = 0
    if "Rotated" not in df.columns:
        df["Rotated"] = 0

    if "Opt Gap" not in df.columns:
        if "True Opt" in df.columns:
            df["Opt Gap"] = (df["Final Loss"] - df["True Opt"]).clip(lower=0)
        else:
            min_found = df.groupby(["Source", "Dimension", "Seed"])["Final Loss"].transform("min")
            df["Opt Gap"] = (df["Final Loss"] - min_found).clip(lower=0)
    else:
        missing_gap = df["Opt Gap"].isna()
        if missing_gap.any() and "True Opt" in df.columns:
            df.loc[missing_gap, "Opt Gap"] = (
                df.loc[missing_gap, "Final Loss"] - df.loc[missing_gap, "True Opt"]
            ).clip(lower=0)

    df["Source"] = df["Source"].astype(str)
    df["Family"] = df["Family"].astype(str)
    df["Optimizer"] = df["Optimizer"].astype(str)
    df["Optimizer_Base"] = df["Optimizer"].apply(get_base_optimizer)
    df["Transform"] = np.where(
        df["Instance"].eq(0),
        "original",
        np.where(df["Rotated"].eq(1), "shift+rotation", "shift"),
    )
    return df


def clean_legacy_problem_name(problem_name: str) -> str:
    return re.sub(r"\s*\(?[\d]+D\)?\s*", "", str(problem_name)).strip()


def get_legacy_problem_family(problem_name: str) -> str:
    if "Linear Regression" in problem_name:
        return "Linear Regression"
    if "BBOB" in problem_name:
        return "BBOB"
    if "ABS" in problem_name or "Layeb" in problem_name or "DIRECTGOLib" in problem_name:
        return "DIRECTGOLib"
    return "Legacy"


def get_legacy_true_optimum(problem_name: str, dimension: int) -> float:
    if "BBOB" in problem_name and ioh is not None:
        match = re.search(r"F(\d+)", problem_name)
        if match:
            fid = int(match.group(1))
            try:
                problem = ioh.get_problem(
                    fid,
                    instance=1,
                    dimension=int(dimension),
                    problem_class=ioh.ProblemClass.BBOB,
                )
                return float(problem.optimum.y)
            except Exception:
                return np.nan
    if "ABS" in problem_name or "Layeb" in problem_name:
        return 0.0
    return np.nan


def read_legacy_results(path: Path) -> tuple[pd.DataFrame, int, str | None]:
    df = pd.read_csv(path, sep=";")
    df.columns = [col.strip().lstrip("\ufeff") for col in df.columns]
    required = {"Problem", "Dimension", "Optimizer", "Seed", "Final Loss", "Total Evals", "Time (s)"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for column in ["Dimension", "Seed", "Final Loss", "Total Evals", "Time (s)"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    directgolib_mask = df["Problem"].astype(str).str.contains(
        r"DIRECTGOLib|ABS|Layeb",
        case=False,
        na=False,
        regex=True,
    )
    excluded_count = int(directgolib_mask.sum())
    df = df.loc[~directgolib_mask].copy()

    df["Source"] = df["Problem"].apply(clean_legacy_problem_name)
    df["Family"] = df["Source"].apply(get_legacy_problem_family)
    df["Instance"] = 0
    df["Rotated"] = 0
    df["Shift Norm"] = 0.0
    df["Transform"] = "legacy"
    df["Dataset"] = "legacy_non_directgolib"
    df["Optimizer"] = df["Optimizer"].astype(str)
    df["Optimizer_Base"] = df["Optimizer"].apply(get_base_optimizer)
    df["Optimizer_Config_Key"] = df["Optimizer"].apply(get_optimizer_config_key)

    df["True Opt"] = df.apply(
        lambda row: get_legacy_true_optimum(row["Source"], row["Dimension"]),
        axis=1,
    )
    min_found = df.groupby(["Source", "Dimension", "Seed"])["Final Loss"].transform("min")
    df["Opt Gap"] = (df["Final Loss"] - df["True Opt"].fillna(min_found)).clip(lower=0)

    warning = None
    if ioh is None and df["Source"].astype(str).str.contains("BBOB", case=False, na=False).any():
        warning = (
            "ioh is not available: BBOB gaps in the legacy CSV use the per-seed empirical "
            "best fallback for configuration selection."
        )
    return df, excluded_count, warning


def select_fair_best_configs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fair_groups = []
    selected_rows = []

    for (dimension, optimizer_base), group in df.groupby(["Dimension", "Optimizer_Base"]):
        config_column = "Optimizer_Config_Key" if "Optimizer_Config_Key" in group.columns else "Optimizer"
        gap_for_selection = group["Opt Gap"].replace([np.inf, -np.inf], np.nan)
        config_performance = group.assign(gap_for_selection=gap_for_selection).groupby(config_column)[
            "gap_for_selection"
        ].mean()
        config_performance = config_performance.fillna(np.inf)
        best_config = config_performance.idxmin()
        fair_groups.append(group[group[config_column] == best_config])
        selected_rows.append(
            {
                "Dimension": dimension,
                "Optimizer_Base": optimizer_base,
                "Selected Optimizer": best_config,
                "Finite Mean Opt Gap": config_performance.loc[best_config],
            }
        )

    return pd.concat(fair_groups, ignore_index=True), pd.DataFrame(selected_rows)


def aggregate_results(df_best: pd.DataFrame) -> pd.DataFrame:
    grouped = [
        "Source",
        "Family",
        "Dimension",
        "Instance",
        "Rotated",
        "Transform",
        "Optimizer_Base",
    ]
    df_agg = (
        df_best.groupby(grouped)
        .agg(
            mu_gap=("Opt Gap", "mean"),
            std_gap=("Opt Gap", "std"),
            mu_time=("Time (s)", "mean"),
            mu_evals=("Total Evals", "mean"),
            n_runs=("Opt Gap", "size"),
            n_failed=("Opt Gap", lambda values: np.isinf(values).sum()),
        )
        .reset_index()
    )
    rank_keys = ["Source", "Dimension", "Instance", "Rotated"]
    df_agg["Rank"] = df_agg.groupby(rank_keys)["mu_gap"].rank(method="min")
    return df_agg


def ordered_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in OPTIMIZER_ORDER if column in table.columns]


def render_table(table: pd.DataFrame) -> str:
    if hasattr(table, "to_markdown"):
        try:
            return table.to_markdown()
        except ImportError:
            pass
    return table.to_string()


def print_pivot(table: pd.DataFrame, title: str, digits: int = 2) -> None:
    cols = ordered_columns(table)
    if cols:
        table = table[cols]
    print(f"\n--- {title} ---")
    print(render_table(table.round(digits)))


def format_gap(mu: float, std: float) -> str:
    if pd.isna(mu):
        return "nan"
    if np.isinf(mu):
        return "inf"
    if pd.isna(std):
        std = 0.0
    if 0 < abs(mu) < 0.01 or abs(mu) >= 1e4 or abs(std) >= 1e4:
        return f"{mu:.2e} ± {std:.2e}"
    return f"{mu:.2f} ± {std:.2f}"


def print_gap_table(df_dim: pd.DataFrame) -> None:
    df_gap = df_dim.copy()
    df_gap["Gap"] = df_gap.apply(lambda row: format_gap(row["mu_gap"], row["std_gap"]), axis=1)
    table = df_gap.pivot_table(
        index="Source",
        columns="Optimizer_Base",
        values="Gap",
        aggfunc="first",
    )
    cols = ordered_columns(table)
    if cols:
        table = table[cols]
    print("\n--- DEEP-DIVE TABLE: Real Point Values (Optimality Gap) ---")
    print(render_table(table))


def print_ablation_table(df_dim: pd.DataFrame) -> None:
    df_ablation = df_dim[df_dim["Optimizer_Base"].isin(["CDS (Ours)", "Pure Grid Search"])]
    if df_ablation.empty:
        return
    table = df_ablation.pivot_table(
        index="Source",
        columns="Optimizer_Base",
        values="mu_gap",
        aggfunc="first",
    )
    cols = [col for col in ["CDS (Ours)", "Pure Grid Search"] if col in table.columns]
    print("\n--- TABLE 3: Ablation Study (Gap) ---")
    print(render_table(table[cols].map(lambda value: f"{value:.4e}" if np.isfinite(value) else "inf")))


def print_failure_summary(df_agg: pd.DataFrame) -> None:
    failures = (
        df_agg.groupby("Optimizer_Base")["n_failed"]
        .sum()
        .loc[lambda values: values > 0]
        .sort_values(ascending=False)
    )
    if failures.empty:
        return
    print("\n--- FAILURE SUMMARY: Infinite/Failed Runs Kept as Worst Gaps ---")
    print(render_table(failures.to_frame("failed_runs")))


def build_combined_time_dataset(direct_df: pd.DataFrame, legacy_csv: Path) -> tuple[pd.DataFrame, int, str | None]:
    legacy_df, excluded_count, warning = read_legacy_results(legacy_csv)

    direct_combined = direct_df.copy()
    direct_combined["Source"] = "DIRECTGOLib " + direct_combined["Source"].astype(str)
    direct_combined["Dataset"] = "directgolib_original"
    direct_combined["Optimizer_Config_Key"] = direct_combined["Optimizer"].apply(get_optimizer_config_key)
    direct_combined["Transform"] = np.where(
        direct_combined["Instance"].eq(0),
        "directgolib_original",
        direct_combined["Transform"],
    )

    common_columns = [
        "Problem",
        "Family",
        "Source",
        "Dimension",
        "Instance",
        "Rotated",
        "Optimizer",
        "Optimizer_Base",
        "Optimizer_Config_Key",
        "Seed",
        "Final Loss",
        "True Opt",
        "Opt Gap",
        "Total Evals",
        "Time (s)",
        "Shift Norm",
        "Transform",
        "Dataset",
    ]
    combined = pd.concat(
        [legacy_df[common_columns], direct_combined[common_columns]],
        ignore_index=True,
    )
    return combined, excluded_count, warning


def print_combined_time_analysis(direct_df: pd.DataFrame, legacy_csv: Path, dimension: int) -> None:
    combined, excluded_count, warning = build_combined_time_dataset(direct_df, legacy_csv)
    df_best, selected = select_fair_best_configs(combined)
    df_agg = aggregate_results(df_best)

    available_dimensions = sorted(df_agg["Dimension"].dropna().astype(int).unique().tolist())
    selected_dimension = dimension if dimension in available_dimensions else available_dimensions[0]
    df_dim = df_agg[df_agg["Dimension"].eq(selected_dimension)].copy()

    print("\n--- COMBINED TIME ANALYSIS ---")
    print(f"Legacy CSV: {legacy_csv}")
    print(f"Excluded legacy DIRECTGOLib/ABS/Layeb rows: {excluded_count:,}")
    print(
        f"Combined sources in {selected_dimension}D: {df_dim['Source'].nunique()} "
        f"({', '.join(sorted(combined['Dataset'].unique()))})"
    )
    if warning:
        print(f"Warning: {warning}")

    selected_table = selected[selected["Dimension"].eq(selected_dimension)].sort_values("Optimizer_Base")
    selected_table = selected_table.set_index(["Dimension", "Optimizer_Base"])
    print("\n--- COMBINED SELECTED CONFIGURATIONS ---")
    print(render_table(selected_table))

    time_table = (
        df_dim.groupby("Optimizer_Base")
        .agg(
            mean_time_s=("mu_time", "mean"),
            std_problem_time_s=("mu_time", "std"),
            n_sources=("Source", "nunique"),
        )
        .sort_values("mean_time_s")
    )
    print(f"\n--- TABLE 5: Combined Mean Wall-Clock Time in {selected_dimension}D (seconds) ---")
    print(render_table(time_table.round(2)))


def main() -> None:
    args = parse_args()
    df = read_results(args.csv)
    df_best, selected = select_fair_best_configs(df)
    df_agg = aggregate_results(df_best)

    available_dimensions = sorted(df_agg["Dimension"].dropna().astype(int).unique().tolist())
    dimension = args.dimension if args.dimension in available_dimensions else available_dimensions[0]
    df_dim = df_agg[df_agg["Dimension"].eq(dimension)].copy()

    print("DIRECTGOLib ABS/Layeb ranking analysis")
    print(f"Input CSV: {args.csv}")
    print(
        f"Rows: {len(df):,} | Sources: {df['Source'].nunique()} | "
        f"Families: {', '.join(sorted(df['Family'].unique()))} | "
        f"Dimensions: {available_dimensions}"
    )
    print(
        "Selection rule: one best hyperparameter configuration per "
        "(Dimension, Optimizer_Base), as in ranking.py."
    )
    print("Non-finite gaps are ignored only for configuration selection and kept in final tables.")

    selected_table = selected.sort_values(["Dimension", "Optimizer_Base"]).set_index(
        ["Dimension", "Optimizer_Base"]
    )
    print("\n--- SELECTED CONFIGURATIONS ---")
    print(render_table(selected_table))

    df_dim["Rank"] = df_dim.groupby(["Source", "Instance", "Rotated"])["mu_gap"].rank(method="min")
    family_rank = df_dim.pivot_table(
        index="Family",
        columns="Optimizer_Base",
        values="Rank",
        aggfunc="mean",
    )
    family_rank.loc[f"OVERALL {dimension}D"] = df_dim.groupby("Optimizer_Base")["Rank"].mean()
    print_pivot(family_rank, f"TABLE 1: Average Rank {dimension}D (By Family)")

    source_rank = df_dim.pivot_table(
        index="Source",
        columns="Optimizer_Base",
        values="Rank",
        aggfunc="mean",
    )
    print_pivot(source_rank, f"TABLE 2: Average Rank {dimension}D (By Source)")

    print_ablation_table(df_dim)

    time_table = df_dim.groupby("Optimizer_Base")["mu_time"].mean().sort_values().to_frame("Mean Time (s)")
    print(f"\n--- TABLE 4: Mean Wall-Clock Time in {dimension}D (seconds) ---")
    print(render_table(time_table.round(2)))

    print_gap_table(df_dim)
    print_failure_summary(df_agg)

    if args.legacy_csv is not None:
        print_combined_time_analysis(df, args.legacy_csv, dimension)


if __name__ == "__main__":
    main()
