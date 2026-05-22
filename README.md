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
python3 main.py hpo-quick
```

## Output
- Benchmark rows are streamed to CSV during execution.
- A summary table is printed at the end (`Best Config per Method`).

## Main Files
- `main.py`: CLI entrypoint
- `benchmarking_module.py`: benchmark pipeline and optimizer wrappers
- `core.py`: CDS implementations
- `hpo_module.py`: HPO benchmark on Fashion-MNIST
