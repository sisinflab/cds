from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence


COMMANDS = {
    "grid-ablation": "analyses.grid_ablation_table",
    "direct-ranking": "analyses.direct_ranking",
    "direct-bias": "analyses.direct_bias",
    "legacy-ranking": "analyses.ranking",
    "pareto": "analyses.pareto",
    "sensitivity": "analyses.sensitivity_analysis",
    "plots": "analyses.plots",
}


def command_help() -> str:
    commands = ", ".join(sorted(COMMANDS))
    return (
        "usage: python -m analyses <command> [options]\n\n"
        "Run benchmark analyses from a single entrypoint.\n\n"
        f"commands: {commands}\n"
    )


def parse_args(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(command_help())
        raise SystemExit(0)

    command = args[0]
    if command not in COMMANDS:
        print(command_help(), file=sys.stderr)
        raise SystemExit(f"unknown analysis command: {command}")

    return command, args[1:]


def main(argv: Sequence[str] | None = None) -> None:
    command, passthrough = parse_args(argv)
    module_name = COMMANDS[command]
    sys.argv = [f"python -m analyses {command}", *passthrough]
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
