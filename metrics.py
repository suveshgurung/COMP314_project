"""
metrics.py
==========

Timing and memory-measurement harness used to turn a single algorithm
call into statistically meaningful performance data.

This module is deliberately algorithm-agnostic: it has no knowledge of
modular exponentiation specifically, and works with any callable of the
shape f(base, exponent, modulus, **kwargs) -> (result, OperationCounter)
-- which is exactly the shared interface every function in
algorithms.py and montgomery.py implements.

The core entry point, measure():

    1. Runs the given callable repeats times (>= 30 by default, per
       the project's statistical requirements) and times each run with
       time.perf_counter (the highest-resolution monotonic clock in
       the standard library).
    2. Separately runs the callable a handful more times under
       tracemalloc to measure peak / average / auxiliary memory
       usage, WITHOUT tracemalloc's own overhead contaminating the
       timing numbers (memory profiling and timing are measured in two
       independent passes).
    3. Aggregates the timing samples into mean / min / max / standard
       deviation.
    4. Packages everything -- including whatever operation counts the
       algorithm itself reported -- into a single, flat-exportable
       BenchmarkResult.

Why tracemalloc instead of resource.getrusage
-----------------------------------------------
tracemalloc measures Python-level object allocations directly and
is portable across Linux/macOS/Windows, unlike resource.ru_maxrss
(Unix-only, and reports whole-process OS-level RSS, which is dominated
by interpreter startup cost rather than the algorithm's own footprint).
Since every algorithm here is pure-Python arithmetic on int objects,
tracemalloc's allocation-level view is the more precise and portable
choice for isolating each algorithm's own memory behavior -- e.g. making
the sliding-window odd-powers table's memory cost actually visible.

Caveat (documented, not hidden): tracemalloc tracks Python object
allocations via pymalloc; it will not see below that layer (e.g. the
libc-level memory used internally by CPython's bignum arithmetic for a
*single* very large int object being resized in place). It is, however,
perfectly suited to what this project needs: comparing *relative*
memory footprints across algorithms that allocate different NUMBERS of
temporary int objects (e.g. sliding window's precomputed table vs. the
O(1)-auxiliary iterative algorithms).
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from algorithms import OperationCounter


# ---------------------------------------------------------------------------
# Structured result data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    """
    Aggregated wall-clock timing statistics across repeated runs of an
    algorithm on IDENTICAL inputs. All times are in seconds.
    """

    mean_seconds: float
    min_seconds: float
    max_seconds: float
    stdev_seconds: float
    repeats: int
    raw_seconds: Tuple[float, ...] = field(repr=False, default_factory=tuple)


@dataclass
class MemoryStats:
    """
    Memory usage statistics, aggregated across a small number of
    dedicated tracemalloc-instrumented runs.

    Attributes
    ----------
    peak_bytes : float
        The single highest peak traced memory observed across the
        dedicated memory-measurement runs.
    average_bytes : float
        The mean of the per-run peak traced memory across those runs.
        Distinct from peak_bytes: peak captures the worst observed
        case, average smooths out run-to-run allocator/GC noise.
    auxiliary_bytes : float
        Memory attributable to the algorithm's OWN extra data structures
        (e.g. sliding window's odd-power table, Montgomery's precomputed
        context), estimated as the mean of (peak - baseline) across the
        dedicated runs, where baseline is memory already traced
        immediately before each call begins.
    """

    peak_bytes: float
    average_bytes: float
    auxiliary_bytes: float


@dataclass
class BenchmarkResult:
    """
    A single, complete benchmark record: one algorithm, evaluated on one
    (base, exponent, modulus) input, aggregated over repeated runs.

    This is the atomic unit that benchmark.py accumulates into a
    list and eventually exports to CSV / JSON / a pandas DataFrame.
    """

    algorithm: str
    base: int
    exponent: int
    modulus: int
    exponent_bits: int
    modulus_bits: int
    timing: TimingStats
    memory: MemoryStats
    operations: OperationCounter
    result_correct: Optional[bool] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten this nested result into a single-level dict, suitable
        for a CSV row, a JSON record, or a pandas DataFrame row.
        """
        return {
            "algorithm": self.algorithm,
            "base": self.base,
            "exponent": self.exponent,
            "modulus": self.modulus,
            "exponent_bits": self.exponent_bits,
            "modulus_bits": self.modulus_bits,
            "mean_time_s": self.timing.mean_seconds,
            "min_time_s": self.timing.min_seconds,
            "max_time_s": self.timing.max_seconds,
            "stdev_time_s": self.timing.stdev_seconds,
            "repeats": self.timing.repeats,
            "peak_memory_bytes": self.memory.peak_bytes,
            "average_memory_bytes": self.memory.average_bytes,
            "auxiliary_memory_bytes": self.memory.auxiliary_bytes,
            "multiplications": self.operations.multiplications,
            "squarings": self.operations.squarings,
            "modulo_operations": self.operations.modulo_operations,
            "montgomery_reductions": self.operations.montgomery_reductions,
            "result_correct": self.result_correct,
        }


# ---------------------------------------------------------------------------
# Memory measurement (isolated from timing measurement)
# ---------------------------------------------------------------------------

def _measure_memory(
    func: Callable[..., Tuple[int, OperationCounter]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    memory_repeats: int,
) -> Tuple[Tuple[int, OperationCounter], MemoryStats]:
    """
    Run func(*args, **kwargs) memory_repeats times, each inside
    its own fresh tracemalloc session, and aggregate peak / average
    / auxiliary memory usage.

    A fresh session per run (start/stop) avoids accumulating unrelated
    allocations across runs and keeps each measurement independent.
    """
    peaks: List[float] = []
    auxiliary: List[float] = []
    last_call_result: Optional[Tuple[int, OperationCounter]] = None

    for _ in range(memory_repeats):
        tracemalloc.start()
        try:
            baseline_bytes, _ = tracemalloc.get_traced_memory()
            last_call_result = func(*args, **kwargs)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peaks.append(float(peak_bytes))
        auxiliary.append(max(float(peak_bytes) - float(baseline_bytes), 0.0))

    memory = MemoryStats(
        peak_bytes=max(peaks),
        average_bytes=statistics.mean(peaks),
        auxiliary_bytes=statistics.mean(auxiliary),
    )

    assert last_call_result is not None  # memory_repeats >= 1 is enforced by caller
    return last_call_result, memory


# ---------------------------------------------------------------------------
# Core measurement entry point
# ---------------------------------------------------------------------------

def measure(
    func: Callable[..., Tuple[int, OperationCounter]],
    base: int,
    exponent: int,
    modulus: int,
    algorithm_name: str,
    repeats: int = 30,
    memory_repeats: int = 5,
    expected_result: Optional[int] = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """
    Benchmark a single modular-exponentiation function on a single
    (base, exponent, modulus) input.

    Parameters
    ----------
    func : Callable
        An algorithm function from algorithms.py / montgomery.py,
        with signature (base, exponent, modulus, **kwargs) ->
        (result, OperationCounter).
    base, exponent, modulus : int
        The inputs to benchmark. Held IDENTICAL across all repeats, so
        that only system noise (not input variation) contributes to the
        measured spread.
    algorithm_name : str
        Human-readable label recorded on the result (e.g.
        "sliding_window (w=4)"), used later as a column/legend value.
    repeats : int, optional
        Number of repeated timing runs (default 30, the project's
        minimum statistical-repeat requirement).
    memory_repeats : int, optional
        Number of repeated memory-measurement runs (default 5). Kept
        smaller than repeats by default because tracemalloc adds
        nontrivial overhead; 5 runs is enough to smooth out allocator
        noise without dominating total benchmark time.
    expected_result : Optional[int]
        If provided (typically pow(base, exponent, modulus),
        computed ONCE by the caller and reused for every algorithm on
        this input), the measured result is compared against it and
        recorded in BenchmarkResult.result_correct.
    **kwargs :
        Extra keyword arguments forwarded to func (e.g.
        window_size for sliding_window_mod_exp).

    Returns
    -------
    BenchmarkResult

    Notes
    -----
    Operation counts (multiplications, squarings, modulo operations,
    Montgomery reductions) are deterministic for a fixed input, so they
    are recorded from a single representative run rather than
    "aggregated" -- there is nothing to aggregate, since every repeat
    performs the exact same sequence of arithmetic operations.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if memory_repeats < 1:
        raise ValueError("memory_repeats must be >= 1")

    args = (base, exponent, modulus)

    # --- Timing pass: clean runs, no memory-profiling overhead ---
    raw_times: List[float] = []
    timing_result: Optional[int] = None
    timing_counter: Optional[OperationCounter] = None

    for _ in range(repeats):
        start = time.perf_counter()
        timing_result, timing_counter = func(*args, **kwargs)
        end = time.perf_counter()
        raw_times.append(end - start)

    timing = TimingStats(
        mean_seconds=statistics.mean(raw_times),
        min_seconds=min(raw_times),
        max_seconds=max(raw_times),
        stdev_seconds=statistics.stdev(raw_times) if len(raw_times) > 1 else 0.0,
        repeats=repeats,
        raw_seconds=tuple(raw_times),
    )

    # --- Memory pass: dedicated tracemalloc-instrumented runs ---
    (memory_result, _memory_counter), memory = _measure_memory(
        func, args, kwargs, memory_repeats
    )

    result_correct: Optional[bool] = None
    if expected_result is not None:
        result_correct = (timing_result == expected_result) and (
            memory_result == expected_result
        )

    return BenchmarkResult(
        algorithm=algorithm_name,
        base=base,
        exponent=exponent,
        modulus=modulus,
        exponent_bits=exponent.bit_length(),
        modulus_bits=modulus.bit_length(),
        timing=timing,
        memory=memory,
        operations=timing_counter if timing_counter is not None else OperationCounter(),
        result_correct=result_correct,
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def flatten_results(results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
    """
    Flatten a list of BenchmarkResult objects into plain dicts, ready
    to be written to CSV/JSON or loaded into a pandas DataFrame.
    """
    return [result.to_flat_dict() for result in results]


# ---------------------------------------------------------------------------
# Quick self-check (run this file directly for a fast sanity demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from algorithms import (
        left_to_right_mod_exp,
        right_to_left_mod_exp,
        sliding_window_mod_exp,
    )

    print("Demonstrating metrics.measure() on a small example...\n")

    b, e, m = 123456789, 2 ** 12 - 1, (1 << 256) - 189  # a fixed, reasonably-sized example
    expected = pow(b, e, m)

    demo_results = [
        measure(left_to_right_mod_exp, b, e, m, "left_to_right",
                repeats=20, memory_repeats=3, expected_result=expected),
        measure(right_to_left_mod_exp, b, e, m, "right_to_left",
                repeats=20, memory_repeats=3, expected_result=expected),
        measure(sliding_window_mod_exp, b, e, m, "sliding_window (w=4)",
                repeats=20, memory_repeats=3, expected_result=expected, window_size=4),
    ]

    header = (
        f"{'algorithm':22s} {'mean_ms':>10s} {'stdev_ms':>10s} "
        f"{'mults':>7s} {'sqrs':>7s} {'peak_B':>10s} {'aux_B':>10s} {'correct':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in demo_results:
        print(
            f"{r.algorithm:22s} "
            f"{r.timing.mean_seconds * 1000:10.4f} "
            f"{r.timing.stdev_seconds * 1000:10.4f} "
            f"{r.operations.multiplications:7d} "
            f"{r.operations.squarings:7d} "
            f"{r.memory.peak_bytes:10.0f} "
            f"{r.memory.auxiliary_bytes:10.0f} "
            f"{str(r.result_correct):>8s}"
        )

    assert all(r.result_correct for r in demo_results), "Correctness check failed!"
    print("\nAll results verified correct against pow(). metrics.py OK.")
