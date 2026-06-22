# Cellular Direct Search

Codebase for benchmarking `Cellular Direct Search (CDS)` (under review) against standard optimizers on BBOB and additional test functions.

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python3 main.py quick
python3 main.py benchmark --budget 5000 --seeds 20 --dims 10 20 30 --out results/benchmark_results.csv
python3 main.py directgolib --budget 5000 --seeds 20 --dims 10 --instances 1 2 3 4 5 --out results/directgolib_abs_layeb_shifted.csv
python3 main.py hpo-quick
```

## Output
- Benchmark rows are streamed to CSV during execution.
- A summary table is printed at the end (`Best Config per Method`).
- `directgolib` runs the append-only Python port of all DIRECTGOLib ABS and Layeb functions with five shifted/rotated instances; the default `benchmark` suite is unchanged.

## Main Files
- `main.py`: CLI entrypoint
- `benchmarking_module.py`: benchmark pipeline and optimizer wrappers
- `directgolib_abs_layeb.py`: isolated ABS/Layeb DIRECTGOLib port and box-constrained benchmark
- `core.py`: CDS implementations
- `hpo_module.py`: HPO benchmark on Fashion-MNIST
