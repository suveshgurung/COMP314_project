"""
main.py
=======

The single command-line entry point for the whole project. Runs the
complete pipeline in three phases:

    1. CORRECTNESS VERIFICATION (utils.run_correctness_tests) -- hundreds
       of random cases across every algorithm, verified against pow().
       Halts immediately if anything is wrong (utils.CorrectnessError).
    2. BENCHMARKING (benchmark.run_benchmark_suite) -- the full
       exponent-size x modulus-size sweep, exported to CSV and JSON.
    3. PLOTTING (plots.generate_all_plots) -- all ten required
       publication-quality figures, generated from the exported CSV.

Each phase can be independently skipped via CLI flags (useful for, e.g.,
re-generating plots from an existing benchmark.csv without re-running the
full sweep). All benchmark-tuning flags from benchmark.py are
available here too (this module's argument parser is built ON TOP OF
benchmark.py's, via argparse's parents= mechanism, so no flag
is ever defined twice).

Typical usage
-------------
    python main.py                                   # full pipeline, default settings
    python main.py --quick                            # fast smoke-test of the whole pipeline
    python main.py --skip-correctness --skip-benchmark --output-dir results
                                                        # just re-plot existing results
    python main.py --multiprocessing --workers 8       # parallelize the benchmark sweep
"""

from __future__ import annotations

import argparse
import os
import sys

import benchmark
import plots
from utils import (
    CorrectnessError,
    get_logger,
    run_correctness_tests,
    set_global_seed,
    timed_phase,
)


BANNER = r"""
================================================================================
 Comparative Analysis of Modular Exponentiation Algorithms
 Theoretical and Experimental Evaluation
================================================================================
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the full pipeline CLI parser: every flag from
    benchmark.build_arg_parser(), plus pipeline-level flags for
    skipping phases and tuning the correctness suite.
    """
    benchmark_parser = benchmark.build_arg_parser(add_help=False)

    parser = argparse.ArgumentParser(
        description="Run the full modular exponentiation comparison pipeline: "
        "correctness verification -> benchmarking -> plot generation.",
        parents=[benchmark_parser],
    )

    parser.add_argument("--skip-correctness", action="store_true",
                         help="Skip the correctness-verification phase.")
    parser.add_argument("--skip-benchmark", action="store_true",
                         help="Skip benchmarking (an existing benchmark.csv in "
                              "--output-dir is required for plotting in this case).")
    parser.add_argument("--skip-plots", action="store_true",
                         help="Skip plot generation.")
    parser.add_argument("--correctness-cases", type=int, default=150,
                         help="Random cases per O(log n) algorithm family in the "
                              "correctness suite (default: 150).")

    return parser


def _run_correctness_phase(args: argparse.Namespace, window_sizes) -> None:
    logger = get_logger()
    with timed_phase("correctness verification", logger):
        report = run_correctness_tests(
            num_random_cases=args.correctness_cases,
            window_sizes=sorted(set(window_sizes) | {1}),  # always include the w=1 degenerate case
            seed=args.seed,
            logger=logger,
        )
    logger.info(report.summary())


def _run_benchmark_phase(
    args: argparse.Namespace, config: benchmark.BenchmarkConfig, csv_path: str, json_path: str
) -> None:
    logger = get_logger()

    cases = benchmark.generate_test_cases(config)
    tasks = benchmark.generate_tasks(config)
    logger.info(
        f"Benchmark grid: {len(cases)} test cases -> {len(tasks)} total "
        f"(algorithm, test case) tasks "
        f"(seed={config.seed}, multiprocessing={config.use_multiprocessing})"
    )

    with timed_phase("benchmark suite", logger):
        results = benchmark.run_benchmark_suite(config, show_progress=not args.no_progress)

    logger.info(f"Collected {len(results)} benchmark results; all correctness checks passed.")

    os.makedirs(args.output_dir, exist_ok=True)
    benchmark.export_csv(results, csv_path)
    benchmark.export_json(results, json_path)
    logger.info(f"Results exported to:\n    {csv_path}\n    {json_path}")


def _run_plotting_phase(csv_path: str, output_dir: str) -> None:
    logger = get_logger()

    if not os.path.exists(csv_path):
        logger.error(
            f"Cannot generate plots: '{csv_path}' does not exist. "
            f"Run the pipeline without --skip-benchmark first."
        )
        raise SystemExit(1)

    with timed_phase("plot generation", logger):
        df = plots.load_results(csv_path)
        plot_paths = plots.generate_all_plots(df, output_dir)

    logger.info(f"Generated {len(plot_paths)} plots in '{output_dir}':")
    for name, path in plot_paths.items():
        logger.info(f"    {name:30s} -> {path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logger = get_logger()
    print(BANNER)

    set_global_seed(args.seed)
    config = benchmark.config_from_args(args)

    csv_path = os.path.join(args.output_dir, "benchmark.csv")
    json_path = os.path.join(args.output_dir, "benchmark.json")

    try:
        # --- Phase 1: correctness verification ---
        if not args.skip_correctness:
            _run_correctness_phase(args, config.window_sizes)
        else:
            logger.info("Skipping correctness verification (--skip-correctness).")

        # --- Phase 2: benchmarking ---
        if not args.skip_benchmark:
            _run_benchmark_phase(args, config, csv_path, json_path)
        else:
            logger.info(
                f"Skipping benchmarking (--skip-benchmark); expecting existing "
                f"results at '{csv_path}'."
            )

        # --- Phase 3: plotting ---
        if not args.skip_plots:
            _run_plotting_phase(csv_path, args.output_dir)
        else:
            logger.info("Skipping plot generation (--skip-plots).")

    except CorrectnessError as e:
        logger.error(f"FATAL -- correctness verification failed: {e}")
        sys.exit(1)
    except RuntimeError as e:
        # Raised by benchmark.run_benchmark_suite if any benchmarked result
        # disagrees with pow() -- a second, independent safety net beyond
        # the dedicated correctness phase.
        logger.error(f"FATAL -- benchmark halted: {e}")
        sys.exit(1)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
