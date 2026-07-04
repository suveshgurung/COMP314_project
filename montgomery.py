"""
montgomery.py
=============

Montgomery multiplication and Montgomery modular exponentiation,
implemented entirely from scratch (no built-in pow(), and -- crucially
-- no expensive % modulus divisions anywhere inside the Montgomery
reduction step itself; only bit-shifts and masks, which is the whole
reason Montgomery arithmetic exists).

Background
----------
All of the algorithms in algorithms.py share one expensive recurring
cost: a modular reduction (% modulus) after every multiplication.
Classical integer division/modulo by an arbitrary N is relatively slow
because N is not related to the machine's native word size or radix.

Montgomery's trick is to represent every number in a transformed
"Montgomery domain" relative to a power-of-two radix R = 2^k (with
R > N and gcd(R, N) = 1, which requires N to be ODD):

    a_bar = a * R mod N        ("Montgomery form" of a)

Multiplying two numbers in this domain and reducing back down (the
"REDC" / Montgomery reduction algorithm) can be done using only:
    * bitwise AND (mod R, since R is a power of two)
    * right-shift (division by R, since R is a power of two)
    * ordinary integer multiplication

with NO division by N. This does not change the asymptotic number of
multiplications needed for exponentiation (still Theta(log E)) -- it
changes the *constant factor* per multiplication, which is why Montgomery
exponentiation "improves practical performance without improving
asymptotic complexity" (a required discussion point in this project's
complexity analysis).

Requirements
------------
Montgomery arithmetic requires an ODD modulus (so that R = 2^k is
coprime to N). This is not a limitation in practice: essentially all
cryptographic moduli (RSA moduli, which are products of two large odd
primes) are odd.

Public API
----------
* mod_inverse(a, m)                  -- modular inverse via the
                                             extended Euclidean algorithm
                                             (implemented from scratch).
* MontgomeryContext                  -- precomputed constants (R, R^2
                                             mod N, N') for a fixed modulus.
* montgomery_reduce(t, ctx, counter) -- the REDC algorithm.
* montgomery_multiply(...)           -- one Montgomery multiplication.
* montgomery_square(...)             -- one Montgomery squaring.
* montgomery_mod_exp(base, exponent, modulus) -- full Montgomery
                                             modular exponentiation,
                                             matching the signature shape
                                             of the algorithms in
                                             algorithms.py.
"""

from dataclasses import dataclass
from typing import Tuple

from algorithms import OperationCounter


# ---------------------------------------------------------------------------
# Modular inverse via the Extended Euclidean Algorithm (from scratch)
# ---------------------------------------------------------------------------

def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Iterative extended Euclidean algorithm.

    Returns (g, x, y) such that:
        a * x + b * y = g = gcd(a, b)

    Implemented iteratively (rather than recursively) to avoid any
    recursion-depth concerns for very large operands (e.g. 2048-bit
    moduli).
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t


def mod_inverse(a: int, m: int) -> int:
    """
    Compute a^(-1) mod m using the extended Euclidean algorithm.

    Deliberately implemented from scratch rather than using Python's
    three-argument pow(a, -1, m) shortcut, to keep every piece of
    number-theoretic machinery in this project auditable and
    self-contained.

    Raises
    ------
    ValueError
        If a and m are not coprime (no inverse exists).
    """
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(
            f"{a} has no inverse modulo {m} (gcd = {g}); Montgomery "
            f"arithmetic requires the modulus to be odd."
        )
    return x % m


# ---------------------------------------------------------------------------
# Montgomery context: precomputed constants for a fixed modulus
# ---------------------------------------------------------------------------

@dataclass
class MontgomeryContext:
    """
    Precomputed constants for Montgomery arithmetic modulo a fixed,
    odd modulus.

    Attributes
    ----------
    modulus : int
        N, the (odd) modulus.
    k : int
        Bit-length of the chosen radix, i.e. R = 2^k.
    r : int
        R = 2^k, chosen as the smallest power of two strictly greater
        than N (guaranteed by using N.bit_length() for k).
    r_mask : int
        R - 1. Used to compute "mod R" via a bitwise AND instead of a
        division.
    r_squared_mod_n : int
        R^2 mod N. Used to convert ordinary integers into Montgomery
        form via a single REDC call.
    n_prime : int
        N' = -N^(-1) mod R. The key precomputed constant used inside
        every REDC call.
    """

    modulus: int
    k: int
    r: int
    r_mask: int
    r_squared_mod_n: int
    n_prime: int

    @classmethod
    def build(cls, modulus: int) -> "MontgomeryContext":
        """
        Construct a MontgomeryContext for the given odd modulus.

        This performs a constant amount of *one-time* setup work
        (independent of the exponent): one modular inverse computation
        and one ordinary modulo reduction to obtain R^2 mod N. This
        setup cost is amortized across every multiplication performed
        during exponentiation, which is why Montgomery exponentiation
        pays off increasingly as the exponent (and hence the number of
        multiplications) grows.
        """
        if modulus <= 0:
            raise ValueError("modulus must be a positive integer")
        if modulus % 2 == 0:
            raise ValueError(
                "Montgomery arithmetic requires an ODD modulus, since the "
                "radix R = 2^k must be coprime to the modulus. Use one of "
                "the classical algorithms in algorithms.py for even moduli."
            )

        # Smallest k such that R = 2^k > modulus.
        # bit_length(N) always satisfies 2^(bit_length(N)-1) <= N < 2^bit_length(N),
        # so R = 2^bit_length(N) is guaranteed to exceed N.
        k = modulus.bit_length()
        r = 1 << k
        r_mask = r - 1

        # One-time, ordinary (division-based) modulo reduction -- this is
        # setup cost paid ONCE per modulus, not once per multiplication.
        r_squared_mod_n = (r * r) % modulus

        n_inv_mod_r = mod_inverse(modulus % r, r)
        n_prime = (r - n_inv_mod_r) % r  # N' = -N^(-1) mod R

        return cls(
            modulus=modulus,
            k=k,
            r=r,
            r_mask=r_mask,
            r_squared_mod_n=r_squared_mod_n,
            n_prime=n_prime,
        )


# ---------------------------------------------------------------------------
# Montgomery reduction (REDC) -- the heart of the algorithm
# ---------------------------------------------------------------------------

def montgomery_reduce(t: int, ctx: MontgomeryContext, counter: OperationCounter) -> int:
    """
    REDC(T): compute T * R^(-1) mod N without ever dividing by N.

    Algorithm (Montgomery, 1985)
    -----------------------------
        m = ((T mod R) * N') mod R      # implemented as (T & r_mask) * n_prime & r_mask
        u = (T + m * N) >> k            # exact division by R (remainder is 0 by
                                         # construction of m), implemented as a
                                         # right-shift since R = 2^k
        return u - N if u >= N else u

    Why no division by N is needed
    -------------------------------
    Both "mod R" and "divide by R" are cheap bit operations because R is
    a power of two -- this is the entire point of Montgomery's method.
    The only "modulo-N-like" work left is the final conditional
    subtraction, which is O(1) and division-free.

    This function increments montgomery_reductions (NOT
    modulo_operations) on the counter, to keep the metric distinct
    from the classical division-based % modulus operations counted
    elsewhere -- they are not the same cost, which is exactly the point
    being measured.
    """
    m = ((t & ctx.r_mask) * ctx.n_prime) & ctx.r_mask
    u = (t + m * ctx.modulus) >> ctx.k

    if u >= ctx.modulus:
        u -= ctx.modulus

    counter.montgomery_reductions += 1
    return u


def _raw_montgomery_product(x: int, y: int, ctx: MontgomeryContext, counter: OperationCounter) -> int:
    """Internal helper: REDC(x * y). Does not touch multiplication/squaring counters."""
    return montgomery_reduce(x * y, ctx, counter)


def montgomery_multiply(
    a_bar: int, b_bar: int, ctx: MontgomeryContext, counter: OperationCounter
) -> int:
    """
    Montgomery-multiply two DISTINCT Montgomery-form operands.

    Given a_bar = a*R mod N and b_bar = b*R mod N, returns
    (a*b)*R mod N -- i.e. the Montgomery form of the ordinary product
    a*b -- via a single REDC call.

    Counted as one multiplication (plus the REDC call's own
    montgomery_reductions increment), mirroring how a single
    "general multiplication" is counted in the classical algorithms.
    """
    counter.multiplications += 1
    return _raw_montgomery_product(a_bar, b_bar, ctx, counter)


def montgomery_square(a_bar: int, ctx: MontgomeryContext, counter: OperationCounter) -> int:
    """
    Montgomery-square a single Montgomery-form operand with itself.

    Kept as a separate function (rather than calling
    montgomery_multiply(a_bar, a_bar, ...)) purely so that it can be
    counted as a squaring rather than a multiplication --
    consistent with the operation-counting convention used throughout
    algorithms.py.
    """
    counter.squarings += 1
    return _raw_montgomery_product(a_bar, a_bar, ctx, counter)


def to_montgomery_form(a: int, ctx: MontgomeryContext, counter: OperationCounter) -> int:
    """
    Convert an ordinary residue a (with 0 <= a < N) into Montgomery
    form: a_bar = a * R mod N, computed as REDC(a * R^2 mod N).
    """
    return montgomery_reduce(a * ctx.r_squared_mod_n, ctx, counter)


def from_montgomery_form(a_bar: int, ctx: MontgomeryContext, counter: OperationCounter) -> int:
    """
    Convert a Montgomery-form value back to an ordinary residue:
    a = a_bar * R^(-1) mod N, computed as REDC(a_bar).
    """
    return montgomery_reduce(a_bar, ctx, counter)


# ---------------------------------------------------------------------------
# Montgomery Modular Exponentiation
# ---------------------------------------------------------------------------

def montgomery_mod_exp(base: int, exponent: int, modulus: int) -> Tuple[int, OperationCounter]:
    """
    Compute (base ** exponent) % modulus using Montgomery exponentiation.

    Structure
    ---------
    1. One-time setup: build a MontgomeryContext for modulus
       (requires modulus to be odd).
    2. Enter the Montgomery domain: convert base and the identity
       element (1) into Montgomery form.
    3. Run the SAME right-to-left binary exponentiation loop used in
       algorithms.right_to_left_mod_exp -- but every multiplication
       and squaring inside the loop is a Montgomery multiplication/
       squaring (REDC-based, division-free) rather than a classical
       % modulus multiplication.
    4. Leave the Montgomery domain: convert the final accumulator back
       to an ordinary residue via one last REDC call.

    Reusing the exact right-to-left bit-scanning structure from
    algorithms.py is intentional: it isolates the ONE variable being
    tested (classical modulo vs. Montgomery reduction) so that any
    runtime difference measured in the benchmarks can be attributed to
    the multiplication/reduction strategy, not to a different
    exponentiation control-flow.

    Complexity
    ----------
    Time:  O(log E) Montgomery multiplications -- identical asymptotic
           complexity to classical binary exponentiation. Montgomery's
           benefit is a smaller constant factor per multiplication (no
           division by N), not a better asymptotic class, which is why
           it is expected to pay off increasingly as N grows large
           (e.g. 1024-2048 bit RSA-scale moduli) but may show LESS
           benefit, or even overhead, for small moduli where the
           one-time context-setup cost is comparable to the savings.
    Space: O(1) auxiliary space beyond the fixed-size MontgomeryContext.

    Parameters
    ----------
    base : int
        The base value.
    exponent : int
        Non-negative integer exponent.
    modulus : int
        A positive, ODD integer modulus.

    Returns
    -------
    Tuple[int, OperationCounter]
        The result of (base ** exponent) % modulus, and an operation
        counter (whose montgomery_reductions field will be nonzero,
        unlike the classical algorithms in algorithms.py).

    Raises
    ------
    ValueError
        If modulus is even (Montgomery arithmetic is undefined in
        that case, since R = 2^k could not be coprime to N).
    """
    counter = OperationCounter()

    if modulus == 1:
        return 0, counter

    if modulus % 2 == 0:
        raise ValueError(
            "montgomery_mod_exp requires an ODD modulus. RSA-style "
            "cryptographic moduli (products of two odd primes) always "
            "satisfy this; for even moduli, use algorithms.py instead."
        )

    ctx = MontgomeryContext.build(modulus)

    base = base % modulus
    counter.modulo_operations += 1  # one classical reduction to normalize the raw input

    if exponent == 0:
        return 1 % modulus, counter

    # --- Enter the Montgomery domain ---
    base_bar = to_montgomery_form(base, ctx, counter)
    result_bar = to_montgomery_form(1, ctx, counter)

    # --- Right-to-left binary exponentiation, entirely within the Montgomery domain ---
    e = exponent
    b_bar = base_bar
    while e > 0:
        if e & 1:
            result_bar = montgomery_multiply(result_bar, b_bar, ctx, counter)
        b_bar = montgomery_square(b_bar, ctx, counter)
        e >>= 1

    # --- Leave the Montgomery domain ---
    result = from_montgomery_form(result_bar, ctx, counter)

    return result, counter


# ---------------------------------------------------------------------------
# Quick self-check (run this file directly for a fast sanity test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import time

    random.seed(time.time())
    print("Running a quick self-check of montgomery.py against pow()...\n")

    failures = 0
    trials = 0

    # --- Random odd moduli of varying bit-lengths ---
    for bit_length in (8, 16, 32, 64, 128, 256):
        for _ in range(30):
            trials += 1
            n = random.getrandbits(bit_length) | 1  # force odd
            n = max(n, 3)
            a = random.randint(0, n * 2)
            e = random.randint(0, 5000)

            expected = pow(a, e, n)  # verification-only use of built-in pow()
            got, counter = montgomery_mod_exp(a, e, n)

            if got != expected:
                failures += 1
                print(f"[FAIL] base={a}, exp={e}, mod={n} (bits={bit_length}) "
                      f"-> got {got}, expected {expected}")

    # --- Explicit edge cases ---
    edge_cases = [(0, 5, 7), (5, 0, 7), (5, 3, 1), (7, 1, 13)]
    for a, e, n in edge_cases:
        trials += 1
        expected = pow(a, e, n)
        got, _ = montgomery_mod_exp(a, e, n)
        if got != expected:
            failures += 1
            print(f"[FAIL] edge case base={a}, exp={e}, mod={n} "
                  f"-> got {got}, expected {expected}")

    if failures == 0:
        print(f"All {trials} trials match pow(). Montgomery implementation OK.")
    else:
        print(f"\n{failures} / {trials} mismatch(es) found.")

    # --- Confirm even moduli are correctly rejected ---
    try:
        montgomery_mod_exp(5, 3, 8)
        print("[FAIL] montgomery_mod_exp should have rejected an even modulus.")
    except ValueError:
        print("Even-modulus rejection works as expected.")
