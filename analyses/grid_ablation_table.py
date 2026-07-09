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
COMPLETE_CSV = DEFAULT_RESULTS_DIR / "cds_benchmarking_complete_results.csv"
DIRECT_ORIGINAL_CSV = DEFAULT_RESULTS_DIR / "directgolib_abs_layeb_original.csv"
DIRECT_SHIFTED_CSV = DEFAULT_RESULTS_DIR / "directgolib_abs_layeb_shifted.csv"

CDS_COMPLETE = "CDS (h=0.25, N=8)"
CDS_DIRECT = "CDS-Box (h=0.25, N=8)"
PURE_GRID = "Pure Grid Search (h=0.25)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CDS vs Pure Grid Search ablation tables from raw benchmark CSVs."
    )
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--complete", type=Path, default=COMPLETE_CSV)
    parser.add_argument("--direct-original", type=Path, default=DIRECT_ORIGINAL_CSV)
    parser.add_argument("--direct-shifted", type=Path, default=DIRECT_SHIFTED_CSV)
    parser.add_argument(
        "--include-layeb01",
        action="store_true",
        help="Include Layeb01, which can overflow on shifted/rotated instances.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print the per-source table used to build the compact medians.",
    )
    return parser.parse_args()


def get_bbob_optimum(problem_name: str, dimension: int) -> float:
    if ioh is None:
        return np.nan

    match = re.search(r"F(\d+)", str(problem_name))
    if not match or "BBOB" not in str(problem_name):
        return np.nan

    fid = int(match.group(1))
    problem = ioh.get_problem(
        fid,
        instance=1,
        dimension=int(dimension),
        problem_class=ioh.ProblemClass.BBOB,
    )
    return float(problem.optimum.y)


def read_complete_results(path: Path, dimension: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df = df[df["Dimension"].eq(dimension)].copy()
    df = df[df["Problem"].str.contains("BBOB", na=False) | df["Problem"].eq("Linear Regression")]

    optima = {
        (problem, dimension): get_bbob_optimum(problem, dimension)
        for problem in df["Problem"].dropna().unique()
        if "BBOB" in str(problem)
    }
    df["True Opt"] = df["Problem"].map(lambda problem: optima.get((problem, dimension), np.nan))

    empirical_best = df.groupby(["Problem", "Dimension", "Seed"])["Final Loss"].transform("min")
    df["Opt Gap"] = (df["Final Loss"] - df["True Opt"].fillna(empirical_best)).clip(lower=0)
    df = df[df["Optimizer"].isin([CDS_COMPLETE, PURE_GRID])].copy()
    df["Source"] = df["Problem"]
    df["Condition"] = np.where(df["Problem"].str.contains("BBOB", na=False), "BBOB", "Linear Regression")
    df["Optimizer Key"] = df["Optimizer"].replace({CDS_COMPLETE: "CDS", PURE_GRID: "Pure Grid"})
    return df


def read_direct_results(path: Path, label: str, dimension: int, include_layeb01: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[df["Problem"].astype(str).ne("Problem")].copy()

    for column in ["Dimension", "Instance", "Rotated", "Seed", "Opt Gap", "Total Evals", "Time (s)"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df["Dimension"].eq(dimension)].copy()
    df = df[df["Optimizer"].isin([CDS_DIRECT, PURE_GRID])].copy()
    if not include_layeb01:
        df = df[~df["Source"].eq("Layeb01")].copy()

    if label == "Original":
        df["Condition"] = "Original"
    else:
        df["Condition"] = np.where(df["Rotated"].eq(1), "Shift+rotation", "Shift only")

    df["Optimizer Key"] = df["Optimizer"].replace({CDS_DIRECT: "CDS", PURE_GRID: "Pure Grid"})
    return df


def source_level_table(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    table = (
        df.groupby(group_columns + ["Source", "Optimizer Key"])["Opt Gap"]
        .mean()
        .unstack()
        .reset_index()
    )
    table["Grid/CDS"] = table["Pure Grid"] / table["CDS"].replace(0, np.nan)
    return table


def aggregate_source_rows(source_rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in source_rows.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        finite_ratio = group["Grid/CDS"].replace([np.inf, -np.inf], np.nan)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n": group["Source"].nunique(),
                "CDS": group["CDS"].replace([np.inf, -np.inf], np.nan).median(),
                "Pure Grid": group["Pure Grid"].replace([np.inf, -np.inf], np.nan).median(),
                "Grid/CDS": finite_ratio.median(),
                "CDS Wins": int((group["Grid/CDS"] > 1).sum()),
                "Win Rate": float((group["Grid/CDS"] > 1).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def format_gap(value: float) -> str:
    if not np.isfinite(value):
        return r"$\infty$"
    if value == 0:
        return r"$\mathbf{0.00}$"
    if abs(value) >= 1e3 or abs(value) < 1e-2:
        return rf"${value:.2e}$"
    return rf"${value:.2f}$"


def format_improvement(value: float) -> str:
    if not np.isfinite(value):
        return "conv."
    return f"{value:.1f}x"


def format_wins(row: pd.Series) -> str:
    return f"{int(row['CDS Wins'])}/{int(row['n'])}"


def print_latex_table(table: pd.DataFrame) -> None:
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(
        r"\caption{CDS versus Pure Grid Search on 10D BBOB, linear regression, and DIRECTGOLib variants. "
        r"Both methods use the same grid spacing ($h=0.25$). Values are median optimality gaps across "
        r"problems/sources after averaging seeds; Grid/CDS values above one favor CDS.}"
    )
    print(r"\label{tab:cds_grid_ablation_full}")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(
        r"\textbf{Benchmark block} & \textbf{$n$} & \textbf{CDS} & "
        r"\textbf{Pure Grid} & \textbf{Grid/CDS} & \textbf{CDS wins} \\"
    )
    print(r"\midrule")
    for _, row in table.iterrows():
        label = row["Benchmark block"]
        print(
            f"{label} & {int(row['n'])} & {format_gap(row['CDS'])} & "
            f"{format_gap(row['Pure Grid'])} & {format_improvement(row['Grid/CDS'])} & "
            f"{format_wins(row)} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_detail_table(table: pd.DataFrame) -> None:
    detail = table.copy()
    detail["Grid/CDS"] = detail["Grid/CDS"].replace([np.inf, -np.inf], np.nan)
    detail = detail.rename(columns={"Pure Grid": "Pure Grid", "CDS": "CDS"})
    detail = detail[["Benchmark block", "Source", "CDS", "Pure Grid", "Grid/CDS"]]
    detail = detail.sort_values(["Benchmark block", "Source"])
    print("\nDetailed source table")
    print(detail.round({"CDS": 4, "Pure Grid": 4, "Grid/CDS": 4}).to_markdown(index=False))


def main() -> None:
    args = parse_args()

    complete = read_complete_results(args.complete, args.dimension)
    complete_sources = source_level_table(complete, ["Condition"])
    complete_sources["Benchmark block"] = complete_sources["Condition"]
    complete_agg = aggregate_source_rows(complete_sources, ["Condition"])
    complete_agg["Benchmark block"] = complete_agg["Condition"]

    direct_original = read_direct_results(args.direct_original, "Original", args.dimension, args.include_layeb01)
    direct_shifted = read_direct_results(args.direct_shifted, "Shifted", args.dimension, args.include_layeb01)
    direct = pd.concat([direct_original, direct_shifted], ignore_index=True)
    direct_sources = source_level_table(direct, ["Family", "Condition"])
    direct_sources["Benchmark block"] = (
        "DIRECTGOLib " + direct_sources["Family"] + " (" + direct_sources["Condition"] + ")"
    )
    direct_agg = aggregate_source_rows(direct_sources, ["Family", "Condition"])
    direct_agg["Benchmark block"] = (
        "DIRECTGOLib " + direct_agg["Family"] + " (" + direct_agg["Condition"] + ")"
    )

    output = pd.concat(
        [
            complete_agg[["Benchmark block", "n", "CDS", "Pure Grid", "Grid/CDS", "CDS Wins", "Win Rate"]],
            direct_agg[["Benchmark block", "n", "CDS", "Pure Grid", "Grid/CDS", "CDS Wins", "Win Rate"]],
        ],
        ignore_index=True,
    )

    order = [
        "BBOB",
        "Linear Regression",
        "DIRECTGOLib ABS (Original)",
        "DIRECTGOLib ABS (Shift only)",
        "DIRECTGOLib ABS (Shift+rotation)",
        "DIRECTGOLib Layeb (Original)",
        "DIRECTGOLib Layeb (Shift only)",
        "DIRECTGOLib Layeb (Shift+rotation)",
    ]
    output["order"] = output["Benchmark block"].map({name: i for i, name in enumerate(order)})
    output = output.sort_values("order").drop(columns="order").reset_index(drop=True)

    print("\nCompact table")
    print(output.round({"CDS": 4, "Pure Grid": 4, "Grid/CDS": 4, "Win Rate": 3}).to_markdown(index=False))
    print("\nLaTeX")
    print_latex_table(output)
    if args.detail:
        detail = pd.concat(
            [
                complete_sources[["Benchmark block", "Source", "CDS", "Pure Grid", "Grid/CDS"]],
                direct_sources[["Benchmark block", "Source", "CDS", "Pure Grid", "Grid/CDS"]],
            ],
            ignore_index=True,
        )
        print_detail_table(detail)


if __name__ == "__main__":
    main()
