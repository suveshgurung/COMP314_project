"""
algorithms.py
=============

Core (non-Montgomery) modular exponentiation algorithms, implemented
entirely from scratch -- no calls to Python's built-in pow(base, exp, mod)
anywhere in the computation path. pow() is reserved exclusively for
correctness verification in utils.py.

Every algorithm in this module computes:

    result = (base ** exponent) mod modulus

and returns a tuple (result, OperationCounter) so that the benchmarking
harness in benchmark.py can record fine-grained operation counts
(multiplications, squarings, modulo reductions) alongside wall-clock time.

Algorithms implemented
-----------------------
1. naive_mod_exp            -- O(exponent) repeated multiplication.
2. binary_mod_exp           -- O(log exponent) recursive square-and-multiply.
3. left_to_right_mod_exp    -- O(log exponent) iterative, MSB-first.
4. right_to_left_mod_exp    -- O(log exponent) iterative, LSB-first.
5. sliding_window_mod_exp   -- O(log exponent) with a configurable window,
                                    trading precomputation + memory for fewer
                                    multiplications.

Montgomery multiplication and Montgomery exponentiation live in a separate
module, montgomery.py, since they require additional machinery (the
Montgomery domain transform, REDC, etc.) and a dedicated reduction counter.

Design notes
------------
* All functions share the same OperationCounter dataclass so that
  downstream code (metrics.py, benchmark.py) can treat every
  algorithm uniformly.
* All functions share the same signature shape:
      f(base: int, exponent: int, modulus: int, ...) -> Tuple[int, OperationCounter]
  which makes them trivially interchangeable inside a benchmarking loop.
* Edge cases (modulus == 1, exponent == 0) are handled explicitly so
  that the algorithms remain mathematically correct for all non-negative
  integer inputs, not just "typical" cryptographic-sized ones.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Callable


# ---------------------------------------------------------------------------
# Shared operation-counting data structure
# ---------------------------------------------------------------------------

@dataclass
class OperationCounter:
    """
    Tracks the low-level arithmetic operations performed by a single
    exponentiation call.

    Attributes
    ----------
    multiplications : int
        Count of "general" modular multiplications (accumulator * base-like
        term), i.e. multiplications that are NOT a value squared with itself.
    squarings : int
        Count of modular squarings (value * value). Kept separate from
        multiplications because squaring can, in principle, be
        implemented more cheaply than general multiplication (e.g. via
        dedicated squaring circuits/algorithms), so distinguishing the two
        gives a more faithful cost model.
    modulo_operations : int
        Count of % modulus reductions applied to intermediate values.
    montgomery_reductions : int
        Count of Montgomery REDC reductions. Always 0 for the classical
        (non-Montgomery) algorithms in this module; populated by
        montgomery.py.
    """

    multiplications: int = 0
    squarings: int = 0
    modulo_operations: int = 0
    montgomery_reductions: int = 0

    def reset(self) -> None:
        """Zero out all counters (useful when reusing a counter instance)."""
        self.multiplications = 0
        self.squarings = 0
        self.modulo_operations = 0
        self.montgomery_reductions = 0

    def total_operations(self) -> int:
        """Total count of all tracked arithmetic operations."""
        return (
            self.multiplications
            + self.squarings
            + self.modulo_operations
            + self.montgomery_reductions
        )

    def as_dict(self) -> Dict[str, int]:
        """Serialize counts to a plain dict (handy for CSV/JSON export)."""
        return {
            "multiplications": self.multiplications,
            "squarings": self.squarings,
            "modulo_operations": self.modulo_operations,
            "montgomery_reductions": self.montgomery_reductions,
        }


# ---------------------------------------------------------------------------
# 1. Naive Modular Exponentiation -- O(exponent)
# ---------------------------------------------------------------------------

def naive_mod_exp(base: int, exponent: int, modulus: int) -> Tuple[int, OperationCounter]:
    """
    Compute (base ** exponent) % modulus by repeated multiplication.

    Algorithm
    ---------
    result = 1
    repeat `exponent` times:
        result = (result * base) % modulus

    Complexity
    ----------
    Time:  O(E) modular multiplications, where E is the *value* of the
           exponent (not its bit length). This is exponential in the
           *bit length* of the exponent, since E = 2^(bit_length(E) - 1)
           in the worst case.
    Space: O(1) auxiliary integers (excluding the fixed-size operands
           themselves).

    This is the deliberately "bad" baseline: for a 20-bit exponent
    (E ~ 10^6) it already requires about a million modular multiplications,
    whereas binary exponentiation needs only ~20. It exists purely so the
    speedup of every other algorithm can be measured against it.

    Parameters
    ----------
    base : int
        The base value.
    exponent : int
        Non-negative integer exponent.
    modulus : int
        Positive integer modulus.

    Returns
    -------
    Tuple[int, OperationCounter]
        The result of (base ** exponent) % modulus, and a counter object
        recording the operations performed.
    """
    counter = OperationCounter()

    if modulus == 1:
        return 0, counter

    result = 1 % modulus
    counter.modulo_operations += 1

    base = base % modulus
    counter.modulo_operations += 1

    for _ in range(exponent):
        result = (result * base) % modulus
        counter.multiplications += 1
        counter.modulo_operations += 1

    return result, counter


# ---------------------------------------------------------------------------
# 2. Binary Modular Exponentiation (recursive square-and-multiply)
# ---------------------------------------------------------------------------

def binary_mod_exp(base: int, exponent: int, modulus: int) -> Tuple[int, OperationCounter]:
    """
    Compute (base ** exponent) % modulus using the classical recursive
    "square-and-multiply" identity:

        base^0       = 1
        base^e       = (base^(e/2))^2            if e is even
        base^e       = base * base^(e-1)         if e is odd

    Complexity
    ----------
    Time:  O(log E) modular multiplications. Each recursive call either
           halves E (even case) or decrements E by 1 immediately followed
           by a halving on the next call (odd case), so the recursion
           depth -- and hence the total work -- is Theta(log2 E).
    Space: O(log E) due to the recursion call stack (this is the key
           practical difference from the iterative left-to-right /
           right-to-left versions below, which use O(1) auxiliary space).

    This version is included specifically to let the report contrast
    *recursive* (top-down, extra stack space) vs *iterative* (bottom-up,
    constant space) implementations of the exact same mathematical
    recurrence.

    Parameters
    ----------
    base : int
        The base value.
    exponent : int
        Non-negative integer exponent.
    modulus : int
        Positive integer modulus.

    Returns
    -------
    Tuple[int, OperationCounter]
        The result and an operation counter.
    """
    counter = OperationCounter()

    if modulus == 1:
        return 0, counter

    base = base % modulus
    counter.modulo_operations += 1

    def helper(b: int, e: int) -> int:
        if e == 0:
            return 1 % modulus
        if e % 2 == 0:
            half_result = helper(b, e // 2)
            squared = (half_result * half_result) % modulus
            counter.squarings += 1
            counter.modulo_operations += 1
            return squared
        else:
            reduced = helper(b, e - 1)
            product = (b * reduced) % modulus
            counter.multiplications += 1
            counter.modulo_operations += 1
            return product

    result = helper(base, exponent)
    return result, counter


# ---------------------------------------------------------------------------
# 3. Left-to-Right Binary Exponentiation (MSB-first, iterative)
# ---------------------------------------------------------------------------

def left_to_right_mod_exp(base: int, exponent: int, modulus: int) -> Tuple[int, OperationCounter]:
    """
    Compute (base ** exponent) % modulus by scanning the binary
    representation of the exponent from the Most Significant Bit (MSB) to
    the Least Significant Bit (LSB).

    Algorithm
    ---------
    result = 1
    for each bit b in bin(exponent), scanned MSB -> LSB:
        result = result^2 % modulus              # always square
        if b == 1:
            result = (result * base) % modulus   # conditionally multiply

    Complexity
    ----------
    Time:  O(log E) modular multiplications: exactly one squaring per bit,
           plus one extra multiplication per set bit (Hamming weight).
           Worst case (all bits set): 2 * log2(E) multiplications.
    Space: O(1) auxiliary space -- purely iterative, no recursion stack.

    This is the "textbook" square-and-multiply exponentiation most
    commonly taught, and the direction (MSB-first) matters: it processes
    the exponent's bits in the same order the number is naturally written,
    which is why it is the more common form found in cryptographic
    references (e.g., RSA implementations).

    Parameters
    ----------
    base : int
        The base value.
    exponent : int
        Non-negative integer exponent.
    modulus : int
        Positive integer modulus.

    Returns
    -------
    Tuple[int, OperationCounter]
        The result and an operation counter.
    """
    counter = OperationCounter()

    if modulus == 1:
        return 0, counter

    if exponent == 0:
        return 1 % modulus, counter

    base = base % modulus
    counter.modulo_operations += 1

    # bin(exponent) yields a string like '0b1011'; strip the '0b' prefix.
    # The remaining string is already ordered MSB -> LSB.
    bits = bin(exponent)[2:]

    result = 1
    for bit in bits:
        result = (result * result) % modulus
        counter.squarings += 1
        counter.modulo_operations += 1

        if bit == "1":
            result = (result * base) % modulus
            counter.multiplications += 1
            counter.modulo_operations += 1

    return result, counter


# ---------------------------------------------------------------------------
# 4. Right-to-Left Binary Exponentiation (LSB-first, iterative)
# ---------------------------------------------------------------------------

def right_to_left_mod_exp(base: int, exponent: int, modulus: int) -> Tuple[int, OperationCounter]:
    """
    Compute (base ** exponent) % modulus by scanning the binary
    representation of the exponent from the Least Significant Bit (LSB)
    to the Most Significant Bit (MSB).

    Algorithm
    ---------
    result = 1
    while exponent > 0:
        if exponent & 1:
            result = (result * base) % modulus   # conditionally multiply
        base = (base * base) % modulus            # always square the base
        exponent >>= 1

    Complexity
    ----------
    Time:  O(log E) modular multiplications -- identical asymptotic and
           near-identical constant-factor cost to the left-to-right
           version (one squaring per bit, one extra multiplication per
           set bit).
    Space: O(1) auxiliary space.

    Left-to-right vs right-to-left
    -------------------------------
    Both variants perform the same *number* of modular multiplications
    for a given exponent (this project's benchmarks confirm this
    experimentally). The difference is structural:
      * Left-to-right squares an *accumulator that already carries partial
        results*, and needs the bits in MSB-first order (readily available
        via bin()).
      * Right-to-left squares the *base* independently of the accumulator,
        and consumes bits in LSB-first order (readily available via
        exponent & 1 / exponent >>= 1), which makes it convenient
        for hardware implementations and for algorithms (like sliding
        window's cousin, "left-to-right" tables aside) where the base's
        powers are precomputed independently of the accumulator.

    Parameters
    ----------
    base : int
        The base value.
    exponent : int
        Non-negative integer exponent.
    modulus : int
        Positive integer modulus.

    Returns
    -------
    Tuple[int, OperationCounter]
        The result and an operation counter.
    """
    counter = OperationCounter()

    if modulus == 1:
        return 0, counter

    result = 1 % modulus
    counter.modulo_operations += 1

    base = base % modulus
    counter.modulo_operations += 1

    while exponent > 0:
        if exponent & 1:
            result = (result * base) % modulus
            counter.multiplications += 1
            counter.modulo_operations += 1

        base = (base * base) % modulus
        counter.squarings += 1
        counter.modulo_operations += 1

        exponent >>= 1

    return result, counter


# ---------------------------------------------------------------------------
# 5. Sliding Window Exponentiation (configurable window size)
# ---------------------------------------------------------------------------

def sliding_window_mod_exp(
    base: int,
    exponent: int,
    modulus: int,
    window_size: int = 4,
) -> Tuple[int, OperationCounter]:
    """
    Compute (base ** exponent) % modulus using fixed-width sliding-window
    exponentiation.

    Algorithm
    ---------
    1. Precompute all odd powers of `base` up to 2^window_size - 1:
           base^1, base^3, base^5, ..., base^(2^k - 1)
       This costs (2^(k-1) - 1) extra multiplications and one extra
       squaring (to get base^2, used to step between consecutive odd
       powers), where k = window_size.
    2. Scan the exponent's bits MSB -> LSB. On a '0' bit, just square the
       accumulator. On a '1' bit, greedily extend a window of length up to
       `window_size` bits (ending on a '1' bit, to avoid wasting window
       capacity on trailing zeros), square the accumulator once per bit in
       the window, then multiply in the precomputed odd power
       corresponding to that window's value.

    Complexity
    ----------
    Time:  O(log E) squarings (exactly one per bit, same as left-to-right),
           but the number of *multiplications* drops from O(log E) in the
           worst case to roughly O(log E / window_size) on average, at the
           cost of a one-time precomputation of O(2^(window_size - 1))
           multiplications.
    Space: O(2^(window_size - 1)) auxiliary space for the precomputed
           odd-power table, versus O(1) for plain left/right-to-left
           exponentiation.

    Trade-off vs plain binary exponentiation
    -----------------------------------------
    Sliding window amortizes the cost of a multiplication over multiple
    bits, which helps most when multiplications are expensive relative to
    squarings (as in RSA-scale cryptography) and when the exponent is
    large enough that the one-time table precomputation is negligible
    compared to the savings. For very small exponents, or for a
    window_size so large that the precomputation dominates, sliding
    window can be *no better than* (or even slightly worse than) plain
    binary exponentiation -- this trade-off is measured directly in the
    benchmark suite by varying `window_size`.

    Parameters
    ----------
    base : int
        The base value.
    exponent : int
        Non-negative integer exponent.
    modulus : int
        Positive integer modulus.
    window_size : int, optional
        Width of the sliding window in bits (default 4, a common practical
        choice). Must be >= 1. window_size == 1 degenerates to exactly the
        left-to-right binary exponentiation algorithm.

    Returns
    -------
    Tuple[int, OperationCounter]
        The result and an operation counter.
    """
    if window_size < 1:
        raise ValueError("window_size must be a positive integer")

    counter = OperationCounter()

    if modulus == 1:
        return 0, counter

    base = base % modulus
    counter.modulo_operations += 1

    if exponent == 0:
        return 1 % modulus, counter

    # --- Step 1: precompute odd powers of base: base^1, base^3, ..., base^(2^k - 1)
    max_odd_power = (1 << window_size) - 1  # 2^window_size - 1
    odd_powers = {1: base}

    base_squared = (base * base) % modulus
    counter.squarings += 1
    counter.modulo_operations += 1

    for power in range(3, max_odd_power + 1, 2):
        odd_powers[power] = (odd_powers[power - 2] * base_squared) % modulus
        counter.multiplications += 1
        counter.modulo_operations += 1

    # --- Step 2: scan exponent bits MSB -> LSB, applying the sliding window
    bits = bin(exponent)[2:]
    n_bits = len(bits)

    result = 1
    i = 0
    while i < n_bits:
        if bits[i] == "0":
            # No window to apply here: just square and move on.
            result = (result * result) % modulus
            counter.squarings += 1
            counter.modulo_operations += 1
            i += 1
        else:
            # Greedily take up to `window_size` bits, then shrink the
            # window from the right until it ends in a '1' bit (so we
            # never "waste" window width on a trailing zero).
            window_end = min(i + window_size, n_bits)
            while bits[window_end - 1] == "0":
                window_end -= 1

            window_bits = bits[i:window_end]
            window_value = int(window_bits, 2)

            # One squaring per bit consumed by the window.
            for _ in range(len(window_bits)):
                result = (result * result) % modulus
                counter.squarings += 1
                counter.modulo_operations += 1

            # One multiplication to fold in the precomputed odd power.
            result = (result * odd_powers[window_value]) % modulus
            counter.multiplications += 1
            counter.modulo_operations += 1

            i = window_end

    return result, counter


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------
#
# Central lookup table used by benchmark.py and main.py to iterate over
# every classical (non-Montgomery) algorithm generically, without having to
# hard-code function names in multiple places. Montgomery's entry is added
# to a combined registry in benchmark.py, since it lives in montgomery.py
# and needs an extra Montgomery-context precomputation step.

ClassicalAlgorithm = Callable[..., Tuple[int, OperationCounter]]

ALGORITHM_REGISTRY: Dict[str, ClassicalAlgorithm] = {
    "naive": naive_mod_exp,
    "binary_recursive": binary_mod_exp,
    "left_to_right": left_to_right_mod_exp,
    "right_to_left": right_to_left_mod_exp,
    "sliding_window": sliding_window_mod_exp,
}


# ---------------------------------------------------------------------------
# Quick self-check (run this file directly for a fast sanity test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import time

    random.seed(time.time())
    print("Running a quick self-check of algorithms.py against pow()...\n")

    failures = 0
    for trial in range(200):
        m = random.randint(2, 10_000)
        b = random.randint(0, m * 2)
        e = random.randint(0, 5000)

        expected = pow(b, e, m)  # verification-only use of built-in pow()
        for name, func in ALGORITHM_REGISTRY.items():
            got, _ = func(b, e, m)
            if got != expected:
                failures += 1
                print(f"[FAIL] {name}: base={b}, exp={e}, mod={m} "
                      f"-> got {got}, expected {expected}")

    if failures == 0:
        print("All algorithms agree with pow() on 200 random trials. OK.")
    else:
        print(f"\n{failures} mismatch(es) found -- see utils.py for the "
              f"full correctness-testing harness.")
