"""
benchmark.py
============

The benchmarking orchestrator. This module is responsible for:

    1. Generating random (base, exponent, modulus) test cases across the
       full required grid of exponent sizes (2^5 .. 2^20) and modulus
       sizes (32 .. 2048 bits).
    2. Running every algorithm (from ``algorithms.py`` and
       ``montgomery.py``) on every test case, via ``metrics.measure()``.
    3. Verifying every single result against ``pow()`` and HALTING
       immediately if any mismatch is found.
    4. Exporting the full results table to CSV and JSON.
    5. Optionally parallelizing the sweep across CPU cores.

Key experimental design decisions (documented, not hidden)
------------------------------------------------------------
1. **Exponent "size" 2^k means a random k-bit exponent, not the literal
   value 2^k.** A literal value like 2^20 has exactly ONE set bit, which
   would make sliding-window's precomputation look artificially wasteful
   and would not reflect a realistic cryptographic exponent. Instead,
   for a target size 2^k we draw an exponent uniformly from
   [2^(k-1), 2^k - 1], i.e., a uniformly random exponent with EXACTLY k
   bits. This preserves both: (a) magnitude Theta(2^k), which is what
   drives the naive algorithm's O(E) runtime, and (b) a realistic,
   non-degenerate bit pattern, which is what drives the binary /
   sliding-window algorithms' multiplication counts.

2. **All generated moduli are forced ODD.** Montgomery exponentiation is
   mathematically undefined for even moduli (the radix R = 2^k must be
   coprime to N). Rather than special-casing Montgomery with a different
   modulus distribution than every other algorithm (which would make the
   comparison unfair), every algorithm in this benchmark is tested on the
   same odd moduli. This mirrors real-world cryptographic practice, where
   RSA-style moduli (products of two odd primes) are always odd anyway.

3. **Naive is capped at a configurable maximum exponent bit-length**
   (``naive_max_exponent_bits``, default 14, i.e. exponent values up to
   ~16384). Naive is O(E); at E = 2^20 (~10^6) repeated 30 times across
   7 modulus sizes, naive alone would dominate total benchmark runtime by
   several orders of magnitude while adding no new information -- its
   O(n) growth trend is already unambiguous well before that point. This
   cap is fully configurable via the CLI (``--naive-max-exponent-bits``)
   for anyone who wants to push it further.

4. **One representative random input per (exponent_bits, modulus_bits)
   cell, repeated `repeats` times for timing statistics.** The project
   requirement is to repeat "every benchmark" >= 30 times to get robust
   mean/min/max/stdev -- this refers to repeating the SAME configuration
   to smooth out system timing noise, not drawing many different random
   inputs per cell. Operation counts are deterministic for a fixed input
   size class (they depend only on bit-length and Hamming weight, both of
   which are controlled), so a single representative draw per cell is
   sufficient and keeps total runtime tractable.

Public API
----------
* ``BenchmarkConfig``       -- all tunable settings for a benchmark run.
* ``generate_test_cases()`` -- builds the (base, exponent, modulus) grid.
* ``run_benchmark_suite()`` -- runs everything, returns a list of
                                ``BenchmarkResult`` (see ``metrics.py``).
* ``export_csv()`` / ``export_json()`` -- persist results to disk.
* ``main()``                -- CLI entry point (also runnable as
                                ``python benchmark.py --help``).
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from tqdm import tqdm

from algorithms import ALGORITHM_REGISTRY, OperationCounter, sliding_window_mod_exp
from metrics import BenchmarkResult, flatten_results, measure
from montgomery import montgomery_mod_exp


# ---------------------------------------------------------------------------
# Random input generation
# ---------------------------------------------------------------------------

def random_exponent_with_bit_length(bit_length: int, rng: random.Random) -> int:
    """
    Draw a uniformly random integer with EXACTLY ``bit_length`` bits
    (i.e., in the range [2^(bit_length-1), 2^bit_length - 1]).

    Special-cased for bit_length <= 1, returning 0 or 1 respectively,
    since "exactly 0 bits" / "exactly 1 bit" are degenerate edge cases.
    """
    if bit_length <= 0:
        return 0
    if bit_length == 1:
        return 1
    low = 1 << (bit_length - 1)
    high = (1 << bit_length) - 1
    return rng.randint(low, high)


def random_odd_modulus(bit_length: int, rng: random.Random) -> int:
    """
    Draw a uniformly random ODD integer with EXACTLY ``bit_length`` bits.

    The top bit is forced to 1 (to guarantee the exact requested bit
    length) and the bottom bit is forced to 1 (to guarantee oddness, a
    requirement for Montgomery arithmetic).
    """
    if bit_length < 2:
        raise ValueError("modulus bit_length must be >= 2 to be a meaningful odd modulus")
    value = rng.getrandbits(bit_length)
    value |= (1 << (bit_length - 1))  # force exact bit length
    value |= 1                        # force odd
    return value


# ---------------------------------------------------------------------------
# Test case generation
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """One (base, exponent, modulus) input, labeled by its size class."""

    modulus_bits: int
    exponent_bits: int
    base: int
    exponent: int
    modulus: int
    expected_result: int  # pow(base, exponent, modulus), computed once, reused everywhere


def generate_test_cases(config: "BenchmarkConfig") -> List[TestCase]:
    """
    Build the full grid of test cases: one per
    (modulus_bits x exponent_bits) combination.

    A single ``random.Random`` instance, seeded once from
    ``config.seed``, is used for ALL draws in a fixed, deterministic
    order (moduli-bits outer loop, exponent-bits inner loop) so that the
    entire benchmark is exactly reproducible given the same seed.
    """
    rng = random.Random(config.seed)
    cases: List[TestCase] = []

    for modulus_bits in config.modulus_bit_lengths:
        for exponent_bits in config.exponent_bit_lengths:
            modulus = random_odd_modulus(modulus_bits, rng)
            exponent = random_exponent_with_bit_length(exponent_bits, rng)
            base = rng.randint(0, modulus - 1)
            expected_result = pow(base, exponent, modulus)  # verification-only builtin use

            cases.append(
                TestCase(
                    modulus_bits=modulus_bits,
                    exponent_bits=exponent_bits,
                    base=base,
                    exponent=exponent,
                    modulus=modulus,
                    expected_result=expected_result,
                )
            )

    return cases


# ---------------------------------------------------------------------------
# Algorithm resolution (name <-> callable), designed to be pickle-friendly
# ---------------------------------------------------------------------------

def algorithm_names(config: "BenchmarkConfig") -> List[str]:
    """
    The full list of algorithm names tested for a given configuration.

    Sliding window gets one entry PER configured window size (e.g.
    "sliding_window_w2", "sliding_window_w4", "sliding_window_w6"), so
    that the window-size trade-off can be plotted directly.
    """
    names = ["naive", "binary_recursive", "left_to_right", "right_to_left"]
    names += [f"sliding_window_w{w}" for w in config.window_sizes]
    names += ["montgomery"]
    return names


def resolve_algorithm(name: str) -> Callable[..., Tuple[int, OperationCounter]]:
    """
    Map an algorithm name (as produced by ``algorithm_names``) to its
    callable.

    This indirection -- passing plain strings through multiprocessing
    task descriptors and resolving them to actual functions INSIDE each
    worker process -- sidesteps any pickling concerns entirely: nothing
    but plain data (str, int) ever needs to cross a process boundary.
    """
    if name == "montgomery":
        return montgomery_mod_exp

    if name.startswith("sliding_window_w"):
        window_size = int(name[len("sliding_window_w"):])
        return functools.partial(sliding_window_mod_exp, window_size=window_size)

    if name in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[name]

    raise ValueError(f"Unknown algorithm name: {name!r}")


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """
    All tunable settings for a single benchmark run.

    Attributes
    ----------
    exponent_bit_lengths : List[int]
        Values of k for which a random k-bit exponent (magnitude ~2^k)
        is tested. Default: 5..20 inclusive, covering the required
        2^5 .. 2^20 range.
    modulus_bit_lengths : List[int]
        Bit-lengths of the (odd) modulus to test. Default: the required
        32, 64, 128, 256, 512, 1024, 2048.
    window_sizes : List[int]
        Sliding-window widths to benchmark as separate algorithm
        variants.
    repeats : int
        Number of timing repeats per (algorithm, test case) pair
        (>= 30 per the project's statistical requirement).
    memory_repeats : int
        Number of dedicated tracemalloc-instrumented repeats per pair.
    naive_max_exponent_bits : int
        Naive is skipped for exponent_bits beyond this threshold (see
        module docstring, design decision #3).
    seed : Optional[int]
        Random seed for full reproducibility. ``None`` means
        non-deterministic (system entropy).
    use_multiprocessing : bool
        If True, distribute tasks across a process pool.
    num_workers : Optional[int]
        Number of worker processes when ``use_multiprocessing`` is True.
        Defaults to ``os.cpu_count()``.
    """

    exponent_bit_lengths: List[int] = field(default_factory=lambda: list(range(5, 21)))
    modulus_bit_lengths: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048]
    )
    window_sizes: List[int] = field(default_factory=lambda: [2, 4, 6])
    repeats: int = 30
    memory_repeats: int = 5
    naive_max_exponent_bits: int = 14
    seed: Optional[int] = 42
    use_multiprocessing: bool = False
    num_workers: Optional[int] = None


def quick_config(seed: Optional[int] = 42) -> BenchmarkConfig:
    """
    A small, fast configuration for smoke-testing the full pipeline
    end-to-end in seconds rather than minutes.

    NOT intended for the final report -- use the default
    ``BenchmarkConfig()`` (or the CLI defaults) for the complete,
    assignment-required sweep.
    """
    return BenchmarkConfig(
        exponent_bit_lengths=[5, 8, 11],
        modulus_bit_lengths=[32, 64, 128],
        window_sizes=[2, 4],
        repeats=10,
        memory_repeats=2,
        naive_max_exponent_bits=11,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTask:
    """
    A single unit of work: one algorithm, evaluated on one test case.

    Deliberately plain-data (str/int only) so that it is trivially
    picklable for the multiprocessing path.
    """

    algorithm_name: str
    base: int
    exponent: int
    modulus: int
    expected_result: int
    repeats: int
    memory_repeats: int


def generate_tasks(config: BenchmarkConfig) -> List[BenchmarkTask]:
    """
    Expand the test-case grid into the full flat list of
    (algorithm, test case) tasks to run, applying the naive-exponent
    cutoff along the way.
    """
    cases = generate_test_cases(config)
    names = algorithm_names(config)

    tasks: List[BenchmarkTask] = []
    for case in cases:
        for name in names:
            if name == "naive" and case.exponent_bits > config.naive_max_exponent_bits:
                continue  # see module docstring, design decision #3

            tasks.append(
                BenchmarkTask(
                    algorithm_name=name,
                    base=case.base,
                    exponent=case.exponent,
                    modulus=case.modulus,
                    expected_result=case.expected_result,
                    repeats=config.repeats,
                    memory_repeats=config.memory_repeats,
                )
            )

    return tasks


def _run_task(task: BenchmarkTask) -> BenchmarkResult:
    """
    Execute a single ``BenchmarkTask`` and return its ``BenchmarkResult``.

    Module-level (not a closure/lambda) so it can be used directly as
    the target of ``ProcessPoolExecutor.submit`` when multiprocessing
    is enabled.
    """
    func = resolve_algorithm(task.algorithm_name)
    return measure(
        func,
        task.base,
        task.exponent,
        task.modulus,
        task.algorithm_name,
        repeats=task.repeats,
        memory_repeats=task.memory_repeats,
        expected_result=task.expected_result,
    )


# ---------------------------------------------------------------------------
# Running the suite (sequential or multiprocessing)
# ---------------------------------------------------------------------------

def _run_tasks_sequential(tasks: List[BenchmarkTask], show_progress: bool) -> List[BenchmarkResult]:
    iterable = tqdm(tasks, desc="Benchmarking", unit="task") if show_progress else tasks
    return [_run_task(task) for task in iterable]


def _run_tasks_parallel(
    tasks: List[BenchmarkTask], num_workers: Optional[int], show_progress: bool
) -> List[BenchmarkResult]:
    max_workers = num_workers or os.cpu_count() or 1
    results: List[BenchmarkResult] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        iterable = (
            tqdm(as_completed(futures), total=len(futures),
                 desc=f"Benchmarking ({max_workers} workers)", unit="task")
            if show_progress
            else as_completed(futures)
        )
        for future in iterable:
            results.append(future.result())

    return results


def run_benchmark_suite(
    config: BenchmarkConfig, show_progress: bool = True
) -> List[BenchmarkResult]:
    """
    Run the complete benchmark suite described by ``config``.

    Every result is verified against ``pow()`` (via ``expected_result``
    threaded through from ``generate_test_cases``); if ANY result fails
    that verification, this function raises ``RuntimeError`` immediately,
    per the project's "stop execution if any implementation fails"
    requirement. This is a safety net alongside (not a replacement for)
    the dedicated, broader correctness sweep in ``utils.py``, which
    should be run BEFORE benchmarking as a fast pre-flight check.

    Returns
    -------
    List[BenchmarkResult]
    """
    tasks = generate_tasks(config)

    if config.use_multiprocessing:
        results = _run_tasks_parallel(tasks, config.num_workers, show_progress)
    else:
        results = _run_tasks_sequential(tasks, show_progress)

    incorrect = [r for r in results if r.result_correct is False]
    if incorrect:
        preview = "\n".join(
            f"  - {r.algorithm}: base={r.base}, exponent={r.exponent}, modulus={r.modulus}"
            for r in incorrect[:10]
        )
        raise RuntimeError(
            f"HALTING: {len(incorrect)} benchmark result(s) failed correctness "
            f"verification against pow(). This should never happen for correctly "
            f"implemented algorithms -- treat this as a critical bug. "
            f"First failures:\n{preview}"
        )

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(results: List[BenchmarkResult], path: str) -> None:
    """Write all results to a CSV file, one row per (algorithm, test case)."""
    rows = flatten_results(results)
    if not rows:
        raise ValueError("No results to export.")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_json(results: List[BenchmarkResult], path: str) -> None:
    """Write all results to a JSON file as a list of flat records."""
    rows = flatten_results(results)
    with open(path, "w") as json_file:
        json.dump(rows, json_file, indent=2)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    ``add_help`` can be set to False so this parser can be reused as a
    ``parents=[...]`` base for a larger parser (e.g. ``main.py``'s, which
    adds pipeline-level flags on top of every benchmark flag defined
    here) without triggering an "-h/--help conflicts" error from having
    two parsers each try to register their own help flag.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark modular exponentiation algorithms across a "
        "grid of exponent sizes and modulus sizes.",
        add_help=add_help,
    )
    parser.add_argument("--min-exponent-bits", type=int, default=5,
                         help="Smallest exponent bit-length to test (default: 5, i.e. ~2^5).")
    parser.add_argument("--max-exponent-bits", type=int, default=20,
                         help="Largest exponent bit-length to test (default: 20, i.e. ~2^20).")
    parser.add_argument("--modulus-bits", type=int, nargs="+",
                         default=[32, 64, 128, 256, 512, 1024, 2048],
                         help="Modulus bit-lengths to test.")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[2, 4, 6],
                         help="Sliding-window widths to benchmark.")
    parser.add_argument("--repeats", type=int, default=30,
                         help="Timing repeats per (algorithm, test case) pair.")
    parser.add_argument("--memory-repeats", type=int, default=5,
                         help="Memory-measurement repeats per pair.")
    parser.add_argument("--naive-max-exponent-bits", type=int, default=14,
                         help="Skip the naive algorithm beyond this exponent bit-length.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility.")
    parser.add_argument("--multiprocessing", action="store_true",
                         help="Distribute tasks across a process pool.")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of worker processes (default: os.cpu_count()).")
    parser.add_argument("--output-dir", type=str, default="results",
                         help="Directory to write benchmark.csv / benchmark.json into.")
    parser.add_argument("--no-progress", action="store_true",
                         help="Disable tqdm progress bars.")
    parser.add_argument("--quick", action="store_true",
                         help="Use a small, fast configuration for smoke-testing.")
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    if args.quick:
        config = quick_config(seed=args.seed)
    else:
        config = BenchmarkConfig(
            exponent_bit_lengths=list(range(args.min_exponent_bits, args.max_exponent_bits + 1)),
            modulus_bit_lengths=list(args.modulus_bits),
            window_sizes=list(args.window_sizes),
            repeats=args.repeats,
            memory_repeats=args.memory_repeats,
            naive_max_exponent_bits=args.naive_max_exponent_bits,
            seed=args.seed,
        )

    # These two flags apply regardless of whether --quick was used, so
    # they are set unconditionally after the base config is built.
    config.use_multiprocessing = args.multiprocessing
    config.num_workers = args.workers
    return config


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    cases = generate_test_cases(config)
    tasks = generate_tasks(config)
    print(
        f"Benchmark configuration:\n"
        f"  Exponent bit-lengths : {config.exponent_bit_lengths}\n"
        f"  Modulus bit-lengths  : {config.modulus_bit_lengths}\n"
        f"  Window sizes         : {config.window_sizes}\n"
        f"  Repeats / memory reps: {config.repeats} / {config.memory_repeats}\n"
        f"  Naive cutoff (bits)  : {config.naive_max_exponent_bits}\n"
        f"  Seed                 : {config.seed}\n"
        f"  Test cases           : {len(cases)}\n"
        f"  Total tasks          : {len(tasks)}\n"
        f"  Multiprocessing      : {config.use_multiprocessing} "
        f"(workers={config.num_workers or os.cpu_count()})\n"
    )

    results = run_benchmark_suite(config, show_progress=not args.no_progress)
    print(f"\nCollected {len(results)} benchmark results. All correctness checks passed.")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "benchmark.csv")
    json_path = os.path.join(args.output_dir, "benchmark.json")
    export_csv(results, csv_path)
    export_json(results, json_path)
    print(f"Results exported to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
