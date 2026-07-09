from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "results" / "cds_benchmarking_complete_results.csv"
DEFAULT_PROBLEMS = ("RastriginRotated", "Rosenbrock", "Linear Regression")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TikZ coordinates for loss/evaluation Pareto plots.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    return parser.parse_args(argv)


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def build_pareto_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Problem", "Dimension", "Optimizer"])
        .agg(avg_loss=("Final Loss", "mean"), avg_evals=("Total Evals", "mean"))
        .reset_index()
    )


def print_tikz_coordinates(table: pd.DataFrame, problem_name: str, dimension: int) -> None:
    subset = table[table["Problem"].str.contains(problem_name, na=False) & table["Dimension"].eq(dimension)]
    print(f"\n% --- {problem_name} {dimension}D ---")
    if subset.empty:
        print("% No matching rows found.")
        return

    cds_points = subset[subset["Optimizer"].str.contains("CDS", na=False)]
    cds_coords = " ".join(
        f"({row.avg_evals:.0f}, {row.avg_loss:.4f})" for row in cds_points.itertuples(index=False)
    )
    print(rf"\addplot[only marks, mark=*, color=blue!80] coordinates {{ {cds_coords} }};")
    print(r"\addlegendentry{CDS}")

    others = subset[~subset["Optimizer"].str.contains("CDS", na=False)]
    if others.empty:
        return

    best_others = others.loc[others.groupby("Optimizer")["avg_loss"].idxmin()]
    for row in best_others.itertuples(index=False):
        print(
            rf"\addplot[only marks, mark=triangle*, color=red] coordinates "
            rf"{{ ({row.avg_evals:.0f}, {row.avg_loss:.4f}) }};"
        )
        print(rf"\addlegendentry{{{row.Optimizer}}}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df = load_results(args.csv)
    table = build_pareto_table(df)
    for problem_name in args.problems:
        print_tikz_coordinates(table, problem_name, args.dimension)


if __name__ == "__main__":
    main()
