"""
utils.py
========

Cross-cutting utilities used throughout the project:

    * A comprehensive, hundreds-of-cases correctness-testing harness that
      verifies EVERY implemented algorithm (all classical algorithms,
      every configured sliding-window width, and Montgomery
      exponentiation) against Python's built-in pow(), and HALTS
      IMMEDIATELY (raising CorrectnessError) on the first mismatch --
      exactly as the project requires.
    * Reproducibility helpers (global random seeding).
    * A lightweight timing context manager for coarse-grained phase
      timing (e.g. "how long did the whole correctness suite take?"),
      distinct from metrics.py's fine-grained, statistically
      repeated per-algorithm timing.
    * Shared console logging setup used by benchmark.py / main.py.
    * Small formatting helpers (human-readable byte counts).

This module deliberately does NOT duplicate benchmark.py's
structured, statistically-repeated performance measurement -- that is
metrics.py's and benchmark.py's job. utils.py answers a
narrower, PRIOR question -- "are all the implementations even correct in
the first place?" -- which should be run once, before any benchmark
numbers are trusted.
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from algorithms import ALGORITHM_REGISTRY, OperationCounter, sliding_window_mod_exp
from montgomery import montgomery_mod_exp


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "modexp") -> logging.Logger:
    """
    Return a configured logger shared across the project's modules.

    Idempotent: calling this repeatedly (e.g. from several modules that
    each import it) never attaches duplicate handlers, so log lines are
    never printed twice.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: Optional[int]) -> None:
    """
    Seed Python's GLOBAL random module for reproducibility in
    interactive use / main.py.

    Note: benchmark.py's test-case generation instantiates and seeds
    its own private random.Random() instance and is unaffected by
    (and does not need) this function; this is provided purely for
    convenience elsewhere (e.g. ad-hoc scripts, this module's own
    correctness harness when no explicit seed is threaded through).
    """
    if seed is not None:
        random.seed(seed)


# ---------------------------------------------------------------------------
# Coarse-grained phase timing
# ---------------------------------------------------------------------------

@contextmanager
def timed_phase(description: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """
    Context manager that logs how long a coarse-grained pipeline phase
    took (e.g. "the full correctness suite", "the full benchmark sweep",
    "plot generation").

    Distinct from metrics.measure(): this reports overall pipeline
    progress to the user/log, not statistically rigorous per-algorithm
    timing data.
    """
    log = logger or get_logger()
    start = time.perf_counter()
    log.info(f"Starting: {description} ...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info(f"Finished: {description} ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def human_readable_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string, e.g. '1.53 KB'."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


# ---------------------------------------------------------------------------
# Correctness testing harness
# ---------------------------------------------------------------------------

class CorrectnessError(RuntimeError):
    """
    Raised the instant any algorithm's result disagrees with Python's
    built-in pow(). Deliberately left UNHANDLED by
    run_correctness_tests so it propagates all the way up to whatever
    called it (typically main.py), halting the entire pipeline --
    per the project's "stop execution if any implementation fails"
    requirement.
    """


@dataclass
class CorrectnessReport:
    """Summary of a completed (i.e., fully-passed) correctness run."""

    total_trials: int
    algorithms_tested: List[str]
    elapsed_seconds: float

    def summary(self) -> str:
        return (
            f"Correctness verification PASSED: {self.total_trials} "
            f"(algorithm, test case) checks across {len(self.algorithms_tested)} "
            f"algorithm variant(s) -- {', '.join(self.algorithms_tested)} -- "
            f"in {self.elapsed_seconds:.2f}s."
        )


AlgoEntry = Tuple[str, Callable[..., Tuple[int, OperationCounter]]]


def _classical_algorithms() -> List[AlgoEntry]:
    """All classical (non-Montgomery) algorithms from the shared registry."""
    return list(ALGORITHM_REGISTRY.items())


def _sliding_window_variants(window_sizes: Sequence[int]) -> List[AlgoEntry]:
    """One (name, callable) pair per configured sliding-window width."""
    variants: List[AlgoEntry] = []
    for w in window_sizes:
        name = f"sliding_window_w{w}"
        # `_w=w` binds the CURRENT value of w as a default argument,
        # avoiding the classic Python late-binding-closure bug where
        # every lambda would otherwise end up referencing the loop
        # variable's FINAL value.
        variants.append((name, lambda b, e, m, _w=w: sliding_window_mod_exp(b, e, m, window_size=_w)))
    return variants


def _check(
    name: str,
    func: Callable[..., Tuple[int, OperationCounter]],
    base: int,
    exponent: int,
    modulus: int,
    trial_log: List[str],
) -> None:
    """
    Run ONE (algorithm, test case) trial and compare against pow().
    Raises CorrectnessError immediately on any mismatch.
    """
    expected = pow(base, exponent, modulus)  # verification-only use of built-in pow()
    got, _ = func(base, exponent, modulus)
    trial_log.append(name)

    if got != expected:
        raise CorrectnessError(
            f"CORRECTNESS FAILURE in '{name}': base={base}, exponent={exponent}, "
            f"modulus={modulus} -> got {got}, expected {expected} (per pow()). "
            f"Halting immediately."
        )


def _edge_cases_even_ok(rng: random.Random, modulus_bits: int) -> List[Tuple[int, int, int]]:
    """
    Edge cases safe for algorithms WITHOUT an oddness requirement, i.e.
    every classical algorithm (but NOT Montgomery).
    """
    m = max(rng.getrandbits(modulus_bits) | (1 << (modulus_bits - 1)), 2)
    return [
        (0, 5, m),                                          # base = 0
        (5, 0, m),                                          # exponent = 0
        (7, 1, m),                                          # exponent = 1
        (rng.randint(0, 1000), rng.randint(0, 1000), 1),    # modulus = 1
        (rng.randint(0, 1000), rng.randint(0, 1000), 2),    # smallest possible modulus
        (m, 5, m),                                          # base == modulus
        (m + 1, 5, m),                                      # base slightly exceeds modulus
    ]


def _edge_cases_odd_only(rng: random.Random, modulus_bits: int) -> List[Tuple[int, int, int]]:
    """Edge cases restricted to ODD moduli, safe for Montgomery too."""
    m = rng.getrandbits(modulus_bits) | (1 << (modulus_bits - 1)) | 1
    return [
        (0, 5, m),
        (5, 0, m),
        (7, 1, m),
        (rng.randint(0, 1000), rng.randint(0, 1000), 1),  # modulus = 1 (special-cased even in Montgomery)
        (m - 1, 5, m),                                     # base = modulus - 1
    ]


def run_correctness_tests(
    num_random_cases: int = 150,
    naive_trials: int = 40,
    max_exponent_bits: int = 256,
    max_modulus_bits: int = 512,
    window_sizes: Sequence[int] = (1, 2, 3, 4, 6, 8),
    seed: Optional[int] = 0,
    logger: Optional[logging.Logger] = None,
) -> CorrectnessReport:
    """
    Run a comprehensive correctness sweep across every implemented
    algorithm, verifying each result against pow(). HALTS
    IMMEDIATELY (raising CorrectnessError) on the first mismatch.

    Structure
    ---------
    1. **O(log n) classical algorithms** (binary_recursive,
       left_to_right, right_to_left, every sliding-window width
       including the degenerate w=1 case) are tested against
       num_random_cases random (base, exponent, modulus) triples of
       EITHER modulus parity, across bit-lengths up to
       max_modulus_bits / max_exponent_bits -- since these
       algorithms are logarithmic, testing at large bit-lengths is
       cheap. A battery of explicit edge cases follows.
    2. **Naive** is tested separately with fewer trials
       (naive_trials) over a much smaller exponent range, since it
       is O(exponent value) and would otherwise dominate the harness's
       runtime for no additional correctness insight.
    3. **Montgomery** is tested with the same random-case machinery as
       the classical algorithms, but restricted to ODD moduli (its
       mathematical requirement), plus its own edge cases -- including
       an explicit check that even moduli are correctly REJECTED.

    Parameters
    ----------
    num_random_cases : int
        Random trials generated for the O(log n) classical algorithms,
        and separately for Montgomery.
    naive_trials : int
        Random trials for the naive algorithm specifically.
    max_exponent_bits, max_modulus_bits : int
        Upper bounds on bit-lengths for the O(log n) algorithms' random cases.
    window_sizes : Sequence[int]
        Sliding-window widths to test (w=1 should behave identically to
        plain left-to-right exponentiation).
    seed : Optional[int]
        Seed for this function's internal random generator (independent
        of the global random module), for reproducible test runs.
    logger : Optional[logging.Logger]
        Logger to report progress to; defaults to get_logger().

    Returns
    -------
    CorrectnessReport
        Returned ONLY if every single trial passed; otherwise a
        CorrectnessError is raised and this function never returns.
    """
    log = logger or get_logger()
    rng = random.Random(seed)
    start = time.perf_counter()
    trial_log: List[str] = []

    classical = _classical_algorithms()
    sliding_variants = _sliding_window_variants(window_sizes)
    # Swap the registry's single default-window "sliding_window" entry
    # for the explicit per-width variants requested for this test run.
    classical = [(n, f) for n, f in classical if n != "sliding_window"] + sliding_variants

    non_naive_classical = [(n, f) for n, f in classical if n != "naive"]
    naive_entry = [(n, f) for n, f in classical if n == "naive"]
    montgomery_entry: List[AlgoEntry] = [("montgomery", montgomery_mod_exp)]

    log.info(
        f"Running correctness tests: {num_random_cases} random cases x "
        f"{len(non_naive_classical)} O(log n) algorithm(s), {naive_trials} cases "
        f"for naive, {num_random_cases} cases for Montgomery, plus edge cases."
    )

    # --- 1. O(log n) classical algorithms: random cases, either parity ---
    for _ in range(num_random_cases):
        modulus_bits = rng.randint(4, max_modulus_bits)
        exponent_bits = rng.randint(1, max_exponent_bits)
        modulus = max(rng.getrandbits(modulus_bits) | (1 << (modulus_bits - 1)), 2)
        exponent = rng.getrandbits(exponent_bits)
        base = rng.randint(0, modulus * 2)

        for name, func in non_naive_classical:
            _check(name, func, base, exponent, modulus, trial_log)

    # --- 2. O(log n) classical algorithms: edge cases ---
    for modulus_bits in (4, 8, 32, 128, 512):
        for base, exponent, modulus in _edge_cases_even_ok(rng, modulus_bits):
            for name, func in non_naive_classical:
                _check(name, func, base, exponent, modulus, trial_log)

    # --- 3. Naive: smaller-scale random cases ---
    for _ in range(naive_trials):
        modulus = rng.randint(2, 2 ** 20)
        exponent = rng.randint(0, 3000)
        base = rng.randint(0, modulus * 2)
        for name, func in naive_entry:
            _check(name, func, base, exponent, modulus, trial_log)

    # --- 4. Naive: edge cases ---
    for base, exponent, modulus in [(0, 5, 7), (5, 0, 7), (5, 3, 1), (7, 1, 13), (5, 2000, 97)]:
        for name, func in naive_entry:
            _check(name, func, base, exponent, modulus, trial_log)

    # --- 5. Montgomery: random ODD-modulus cases ---
    for _ in range(num_random_cases):
        modulus_bits = rng.randint(4, max_modulus_bits)
        exponent_bits = rng.randint(1, max_exponent_bits)
        modulus = rng.getrandbits(modulus_bits) | (1 << (modulus_bits - 1)) | 1
        exponent = rng.getrandbits(exponent_bits)
        base = rng.randint(0, modulus * 2)

        for name, func in montgomery_entry:
            _check(name, func, base, exponent, modulus, trial_log)

    # --- 6. Montgomery: edge cases (odd moduli only) ---
    for modulus_bits in (4, 8, 32, 128, 512):
        for base, exponent, modulus in _edge_cases_odd_only(rng, modulus_bits):
            for name, func in montgomery_entry:
                _check(name, func, base, exponent, modulus, trial_log)

    # --- 7. Montgomery: confirm even moduli are correctly REJECTED ---
    try:
        montgomery_mod_exp(5, 3, 8)
    except ValueError:
        pass
    else:
        raise CorrectnessError(
            "Montgomery exponentiation should reject even moduli but did not. "
            "Halting immediately."
        )

    elapsed = time.perf_counter() - start
    algorithms_tested = sorted(
        {name for name, _ in non_naive_classical + naive_entry + montgomery_entry}
    )

    report = CorrectnessReport(
        total_trials=len(trial_log),
        algorithms_tested=algorithms_tested,
        elapsed_seconds=elapsed,
    )
    log.info(report.summary())
    return report


# ---------------------------------------------------------------------------
# Quick self-check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger = get_logger()

    with timed_phase("full correctness suite", logger):
        report = run_correctness_tests()

    print()
    print(report.summary())

    # Demonstrate that the harness actually halts on a real mismatch,
    # by intentionally feeding it a broken "algorithm".
    print("\nDemonstrating halt-on-failure behavior with a deliberately broken function...")

    def _broken_mod_exp(base: int, exponent: int, modulus: int):
        return (pow(base, exponent, modulus) + 1) % modulus, OperationCounter()

    trial_log: List[str] = []
    try:
        _check("broken_demo", _broken_mod_exp, 5, 3, 100, trial_log)
        print("[FAIL] Expected a CorrectnessError but none was raised.")
    except CorrectnessError as e:
        print(f"Correctly raised CorrectnessError:\n  {e}")
