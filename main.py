import argparse
import pandas as pd

# Importiamo tutto dal tuo modulo di benchmarking
from benchmarking_module import (
    BenchmarkSettings,
    create_summary_table,
    run_full_benchmark
)

# Importiamo il modulo HPO
from hpo_module import main as hpo_main, run_quick_benchmark as run_hpo_quick_benchmark


def quick_run() -> pd.DataFrame:
    """Run veloce per testare che tutti gli algoritmi e le librerie funzionino senza errori."""
    print("=====================================================")
    print(" AVVIO QUICK RUN (Test di integrità)")
    print(" Budget: 1000 | Seeds: 1 | Dimensioni: 2D")
    print("=====================================================\n")

    settings = BenchmarkSettings(
        budget_evaluations=1000,
        seeds=(42,),
        dimensions=(2,),
        # Attiviamo tutto per assicurarci che non ci siano crash di importazione
        include_cds=True,
        include_cmaes=True,
        include_pso=True,
        include_de=True,
        include_random=True,
        include_neldermead=True,
        include_pdfo=True,
        include_grid_search=True,
        include_bads=True,
        include_nomad=True,
        include_lshade=True
    )

    results_df = run_full_benchmark(settings)
    create_summary_table(results_df)
    return results_df


def main_cli():
    parser = argparse.ArgumentParser(
        description="Cellular Direct Search (CDS) - Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- MODALITÀ DI ESECUZIONE ---
    parser.add_argument(
        "mode", nargs="?", default="benchmark",
        choices=["benchmark", "quick", "hpo", "hpo-quick"],
        help="Modalità di esecuzione dello script"
    )

    # --- IMPOSTAZIONI GLOBALI ---
    parser.add_argument("--budget", type=int, default=5000, help="Budget totale di valutazioni della Loss")
    parser.add_argument("--seeds", type=int, default=20, help="Numero di independent restarts (seed diversi)")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20, 30, 40, 50], help="Dimensioni dei problemi da testare")
    parser.add_argument("--out", type=str, default="risultati_benchmark.csv",
                        help="Nome del file CSV per il salvataggio")

    # --- FLAG PER DISABILITARE SINGOLI ALGORITMI (Ablation/Parallelizzazione) ---
    group = parser.add_argument_group("Disattivazione Modelli (Usa questi flag per SALTARE algoritmi specifici)")
    group.add_argument("--skip-cds", action="store_true", help="Disabilita Cellular Direct Search")
    group.add_argument("--skip-cmaes", action="store_true", help="Disabilita CMA-ES")
    group.add_argument("--skip-pso", action="store_true", help="Disabilita PSO")
    group.add_argument("--skip-de", action="store_true", help="Disabilita Differential Evolution")
    group.add_argument("--skip-random", action="store_true", help="Disabilita Random Search")
    group.add_argument("--skip-neldermead", action="store_true", help="Disabilita Nelder-Mead (SciPy)")
    group.add_argument("--skip-pdfo", action="store_true", help="Disabilita Powell (PDFO)")
    group.add_argument("--skip-grid", action="store_true", help="Disabilita Pure Grid Search")
    group.add_argument("--skip-bads", action="store_true", help="Disabilita BADS (Consigliato per run veloci)")
    group.add_argument("--skip-nomad", action="store_true", help="Disabilita NOMAD")
    group.add_argument("--skip-lshade", action="store_true", help="Disabilita L-SHADE")
    # group.add_argument("--skip-turbo", action="store_true", help="Disabilita Turbo")

    args = parser.parse_args()

    if args.mode == "benchmark":
        print("=====================================================")
        print(" AVVIO BENCHMARK UFFICIALE (IEEE Access Revision)")
        print("=====================================================")
        print(f" Dimensioni testate : {args.dims}")
        print(f" Budget valutazioni : {args.budget}")
        print(f" Numero di Seeds    : {args.seeds} (da 0 a {args.seeds - 1})")
        print(f" File di Output     : {args.out}")
        print("=====================================================\n")

        # Mappiamo gli argomenti della CLI nei setting del nostro modulo
        # Se un utente passa --skip-bads, args.skip_bads è True, quindi include_bads diventa False.
        settings_kwargs = {
            "budget_evaluations": args.budget,
            "seeds": tuple(range(args.seeds)),
            "dimensions": tuple(args.dims),

            "include_cds": not args.skip_cds,
            "include_cmaes": not args.skip_cmaes,
            "include_pso": not args.skip_pso,
            "include_de": not args.skip_de,
            "include_random": not args.skip_random,
            "include_neldermead": not args.skip_neldermead,
            "include_pdfo": not args.skip_pdfo,
            "include_grid_search": not args.skip_grid,
            "include_bads": not args.skip_bads,
            "include_nomad": not args.skip_nomad,
            "include_lshade": not args.skip_lshade,
            # "include_turbo": not args.skip_turbo
        }

        # Inizializza i settaggi
        settings = BenchmarkSettings(**settings_kwargs)

        # Lancia il super-run
        results_df = run_full_benchmark(settings)

        # Salva su disco (FONDAMENTALE)
        results_df.to_csv(args.out, index=False)
        print(f"\n[+] RUN COMPLETATA! Dati salvati con successo in '{args.out}'.")

        # Mostra la tabella riassuntiva
        create_summary_table(results_df)

    elif args.mode == "quick":
        quick_run()

    elif args.mode == "hpo":
        hpo_main()

    elif args.mode == "hpo-quick":
        run_hpo_quick_benchmark()


if __name__ == "__main__":
    main_cli()