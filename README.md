# Cellular Direct Search

Codebase for benchmarking `Cellular Direct Search (CDS)` against standard optimizers on BBOB, DIRECTGOLib, and additional test functions.

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
python3 main.py directgolib --budget 5000 --seeds 20 --dims 10 --instances 0 --out results/directgolib_abs_layeb_original.csv
python3 main.py directgolib --budget 5000 --seeds 20 --dims 10 --instances 1 2 3 4 5 --out results/directgolib_abs_layeb_shifted.csv
python3 main.py directgolib --budget 5000 --seeds 20 --dims 10 --families Layeb --start-source Layeb02 --out results/directgolib_layeb02_onward.csv
python3 main.py hpo-quick
```

## Analysis
All analysis scripts are exposed through one module entrypoint:

```bash
python3 -m analyses legacy-ranking
python3 -m analyses direct-ranking
python3 -m analyses direct-bias
python3 -m analyses grid-ablation
python3 -m analyses grid-ablation --detail
python3 -m analyses pareto
python3 -m analyses plots --budget 5000 --seed 42
python3 -m analyses sensitivity
```

The default analysis inputs are expected under `results/`:

- `cds_benchmarking_complete_results.csv`
- `directgolib_abs_layeb_original.csv`
- `directgolib_abs_layeb_shifted.csv`

## Output
- Benchmark rows are streamed to CSV during execution.
- A summary table is printed at the end (`Best Config per Method`).
- `directgolib` runs the append-only Python port of all DIRECTGOLib ABS and Layeb functions; instance `0` is original, `1-2` are shifted, and `3-5` are shifted+rotated.
- Raw result files and figure outputs are ignored by Git.

## Main Files
- `main.py`: CLI entrypoint
- `benchmarking_module.py`: benchmark pipeline and optimizer wrappers
- `directgolib_abs_layeb.py`: isolated ABS/Layeb DIRECTGOLib port and box-constrained benchmark
- `core.py`: CDS implementations
- `hpo_module.py`: HPO benchmark on Fashion-MNIST
- `analyses/`: ranking, ablation, bias, Pareto, convergence, and sensitivity analyses
