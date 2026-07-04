"""
plots.py
========

Generates the ten publication-quality visualizations required by the
project, from a benchmark results table (either a pandas DataFrame
already in memory, or a CSV file as produced by benchmark.export_csv).

Design principles applied uniformly across every figure
----------------------------------------------------------
* **Consistent per-algorithm styling.** Every algorithm is assigned the
  SAME color/marker combination in every single figure (see
  _style_for), so a reader can visually track, say, "sliding_window
  (w=6)" across all ten plots without re-reading each legend from
  scratch.
* **Deliberate scale choices, not defaults.** Linear vs. logarithmic
  axes are chosen per-plot based on what needs to be visually legible
  (see each function's docstring for the specific reasoning) rather than
  applying log scale everywhere out of habit.
* **High-resolution output.** Every figure is saved at 300 DPI with a
  tight bounding box, suitable for direct inclusion in a PDF report.
* **Every figure has:** a descriptive title, labeled axes (with units),
  a legend, and a grid.

Public API
----------
* load_results(csv_path)          -- load a benchmark CSV into a
                                          pandas DataFrame.
* plot_* functions (x10)          -- one per required visualization,
                                          each returning the saved file
                                          path.
* generate_all_plots(df, out_dir) -- convenience driver that runs
                                          every plot function and returns
                                          a dict of {name: path}.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; no display server required

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)

plt.rcParams.update(
    {
        "savefig.dpi": FIGURE_DPI,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "axes.axisbelow": True,
    }
)

# Preferred legend/color ordering for the algorithms this project always
# produces; any unrecognized names (e.g. a custom sliding-window width)
# are appended afterward in sorted order, so legends stay stable and
# predictable across every figure.
_PREFERRED_ORDER = [
    "naive",
    "binary_recursive",
    "left_to_right",
    "right_to_left",
    "sliding_window_w2",
    "sliding_window_w4",
    "sliding_window_w6",
    "montgomery",
]

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
_CMAP = plt.get_cmap("tab10")


def _algorithm_order(algorithms: Iterable[str]) -> List[str]:
    """Stable, human-friendly ordering of algorithm names for legends."""
    present = set(algorithms)
    ordered = [a for a in _PREFERRED_ORDER if a in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def _style_for(algorithm: str, index: int) -> Dict[str, object]:
    """
    A consistent color / marker / linestyle for a given algorithm,
    reused across every plot in this module. naive is always drawn
    dashed, since it is the baseline being compared against everywhere.
    """
    return {
        "color": _CMAP(index % 10),
        "marker": _MARKERS[index % len(_MARKERS)],
        "linestyle": "--" if algorithm == "naive" else "-",
        "linewidth": 2,
        "markersize": 6,
    }


def _save(fig: plt.Figure, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    """
    Load a benchmark results CSV (as produced by
    benchmark.export_csv) into a pandas DataFrame, coercing the
    result_correct column to proper booleans.
    """
    df = pd.read_csv(csv_path)
    if "result_correct" in df.columns:
        df["result_correct"] = (
            df["result_correct"].astype(str).str.lower().map({"true": True, "false": False})
        )
    return df


# ---------------------------------------------------------------------------
# Plot 1: Runtime vs Exponent Size
# ---------------------------------------------------------------------------

def plot_runtime_vs_exponent_size(
    df: pd.DataFrame, output_dir: str, modulus_bits: Optional[int] = None
) -> str:
    """
    Plot 1: Runtime vs Exponent Size.

    For a fixed modulus bit-length (default: the LARGEST tested, since
    algorithmic differences are most pronounced there), plots mean
    runtime against exponent bit-length for every algorithm.

    Axes: linear x (exponent bit-length), logarithmic y (runtime).
    On these axes, naive -- whose true cost is Theta(2^bits) since bits
    IS the exponent's bit-length -- appears as a STRAIGHT LINE, because
    log(c * 2^bits) = bits*log(2) + log(c) is linear in bits. Every
    other algorithm here is Theta(bits) and appears markedly CONCAVE
    (flattening out) by comparison. This single plot is the clearest
    visual evidence in the whole report for "naive is exponential in
    the exponent's bit-length (linear in its value); everything else is
    logarithmic in its value."
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, algo in enumerate(_algorithm_order(subset["algorithm"].unique())):
        algo_df = subset[subset["algorithm"] == algo].sort_values("exponent_bits")
        if algo_df.empty:
            continue
        ax.plot(
            algo_df["exponent_bits"], algo_df["mean_time_s"], label=algo, **_style_for(algo, i)
        )

    ax.set_yscale("log")
    ax.set_xlabel("Exponent bit-length  (exponent magnitude $\\approx 2^{bits}$)")
    ax.set_ylabel("Mean runtime, seconds (log scale)")
    ax.set_title(f"Runtime vs. Exponent Size  (modulus = {modulus_bits}-bit)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "runtime.png")


# ---------------------------------------------------------------------------
# Plot 2: Runtime vs Modulus Size
# ---------------------------------------------------------------------------

def plot_runtime_vs_modulus_size(
    df: pd.DataFrame, output_dir: str, exponent_bits: Optional[int] = None
) -> str:
    """
    Plot 2: Runtime vs Modulus Size.

    For a fixed exponent bit-length (default: the LARGEST tested), plots
    mean runtime against modulus bit-length for every algorithm.

    Axes: log-log. The tested modulus sizes (32, 64, ..., 2048) form a
    doubling sequence, so a log-x axis spaces them evenly; a log-y axis
    turns any polynomial relationship time ~ modulus_bits^p into a
    straight line of slope p, which is the natural way to visually
    estimate how each algorithm's per-multiplication cost scales with
    operand size (e.g., whether the underlying bignum multiplication
    behaves closer to schoolbook O(b^2) or a faster sub-quadratic
    algorithm at these sizes).
    """
    if exponent_bits is None:
        exponent_bits = int(df["exponent_bits"].max())

    subset = df[df["exponent_bits"] == exponent_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, algo in enumerate(_algorithm_order(subset["algorithm"].unique())):
        algo_df = subset[subset["algorithm"] == algo].sort_values("modulus_bits")
        if algo_df.empty:
            continue
        ax.plot(
            algo_df["modulus_bits"], algo_df["mean_time_s"], label=algo, **_style_for(algo, i)
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ticks = sorted(subset["modulus_bits"].unique())
    if ticks:
        ax.set_xticks(ticks)
    ax.set_xlabel("Modulus bit-length (log scale)")
    ax.set_ylabel("Mean runtime, seconds (log scale)")
    ax.set_title(f"Runtime vs. Modulus Size  (exponent $\\approx 2^{{{exponent_bits}}}$)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "runtime_vs_modulus_size.png")


# ---------------------------------------------------------------------------
# Plot 3: Number of Multiplications
# ---------------------------------------------------------------------------

def plot_multiplication_counts(
    df: pd.DataFrame, output_dir: str, modulus_bits: Optional[int] = None
) -> str:
    """
    Plot 3: Number of Multiplications vs Exponent Size.

    Logarithmic y-axis so that naive's O(exponent value) multiplication
    count and the O(log exponent) algorithms' counts can share one
    figure without the latter being crushed to an invisible flat line.
    This is the plot that makes sliding window's trade-off concrete:
    at small exponent sizes its precomputation overhead can make it use
    MORE multiplications than plain binary exponentiation; the lines
    cross once the exponent is large enough to amortize that table.
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, algo in enumerate(_algorithm_order(subset["algorithm"].unique())):
        algo_df = subset[subset["algorithm"] == algo].sort_values("exponent_bits")
        if algo_df.empty:
            continue
        ax.plot(
            algo_df["exponent_bits"], algo_df["multiplications"], label=algo, **_style_for(algo, i)
        )

    ax.set_yscale("log")
    ax.set_xlabel("Exponent bit-length")
    ax.set_ylabel("Number of multiplications (log scale)")
    ax.set_title(f"Multiplication Count vs. Exponent Size  (modulus = {modulus_bits}-bit)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "multiplications.png")


# ---------------------------------------------------------------------------
# Plot 4: Number of Squarings
# ---------------------------------------------------------------------------

def plot_squaring_counts(
    df: pd.DataFrame, output_dir: str, modulus_bits: Optional[int] = None
) -> str:
    """
    Plot 4: Number of Squarings vs Exponent Size.

    All binary-style algorithms (left-to-right, right-to-left, sliding
    window, Montgomery) square their accumulator/base exactly once per
    bit, so their squaring counts should overlay almost perfectly --
    this plot is direct visual confirmation of that theoretical claim.
    Naive never squares at all (it only ever multiplies by the base),
    so its series is identically zero and simply does not appear on
    this logarithmic-scale plot -- itself a meaningful (if implicit)
    data point.
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, algo in enumerate(_algorithm_order(subset["algorithm"].unique())):
        algo_df = subset[subset["algorithm"] == algo].sort_values("exponent_bits")
        if algo_df.empty:
            continue
        ax.plot(
            algo_df["exponent_bits"], algo_df["squarings"], label=algo, **_style_for(algo, i)
        )

    ax.set_yscale("log")
    ax.set_xlabel("Exponent bit-length")
    ax.set_ylabel("Number of squarings (log scale)")
    ax.set_title(f"Squaring Count vs. Exponent Size  (modulus = {modulus_bits}-bit)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "squarings.png")


# ---------------------------------------------------------------------------
# Plot 5: Number of Modulo Operations
# ---------------------------------------------------------------------------

def plot_modulo_operation_counts(
    df: pd.DataFrame, output_dir: str, modulus_bits: Optional[int] = None
) -> str:
    """
    Plot 5: Number of Modulo Operations vs Exponent Size.

    Includes naive deliberately (unlike plots 3/4): the dramatic gap
    between naive's O(exponent value) division-based reductions and
    every other algorithm's O(log exponent) reductions is exactly the
    justification for binary exponentiation existing at all, and this
    plot makes that gap visually explicit on a log scale. Montgomery's
    curve sits at a nearly-constant, very low value (just the one
    input-normalization reduction), since its per-multiplication
    reductions are counted separately as `montgomery_reductions` (see
    Plot 6) rather than as classical `modulo_operations`.
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, algo in enumerate(_algorithm_order(subset["algorithm"].unique())):
        algo_df = subset[subset["algorithm"] == algo].sort_values("exponent_bits")
        if algo_df.empty:
            continue
        ax.plot(
            algo_df["exponent_bits"], algo_df["modulo_operations"], label=algo, **_style_for(algo, i)
        )

    ax.set_yscale("log")
    ax.set_xlabel("Exponent bit-length")
    ax.set_ylabel("Number of classical modulo (%) operations (log scale)")
    ax.set_title(f"Modulo Operation Count vs. Exponent Size  (modulus = {modulus_bits}-bit)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "modulo_operations.png")


# ---------------------------------------------------------------------------
# Plot 6: Number of Montgomery Reductions
# ---------------------------------------------------------------------------

def plot_montgomery_reductions(
    df: pd.DataFrame,
    output_dir: str,
    modulus_bits: Optional[int] = None,
    reference_algorithm: str = "right_to_left",
) -> str:
    """
    Plot 6: Number of Montgomery Reductions vs Exponent Size.

    Overlays Montgomery's REDC-call count against a reference classical
    algorithm's `modulo_operations` count (default: right_to_left, which
    shares Montgomery's exact bit-scanning structure -- see
    montgomery.py's docstring). The two curves track each other
    closely in SHAPE (both are Theta(log exponent), roughly one
    reduction per multiplication/squaring), which is precisely the
    point: Montgomery does not reduce the NUMBER of reduction-like
    operations, it reduces the COST of each one (shifts/masks instead of
    division) -- an improvement to the constant factor, not the
    asymptotic operation count.
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    mont_df = subset[subset["algorithm"] == "montgomery"].sort_values("exponent_bits")
    if not mont_df.empty:
        ax.plot(
            mont_df["exponent_bits"],
            mont_df["montgomery_reductions"],
            label="montgomery : Montgomery reductions (REDC calls)",
            color=_CMAP(7),
            marker="o",
            linewidth=2,
            markersize=6,
        )

    ref_df = subset[subset["algorithm"] == reference_algorithm].sort_values("exponent_bits")
    if not ref_df.empty:
        ax.plot(
            ref_df["exponent_bits"],
            ref_df["modulo_operations"],
            label=f"{reference_algorithm} : classical modulo operations",
            color=_CMAP(3),
            marker="s",
            linestyle="--",
            linewidth=2,
            markersize=6,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Exponent bit-length")
    ax.set_ylabel("Operation count (log scale)")
    ax.set_title(
        f"Montgomery Reductions vs. Classical Modulo Operations  (modulus = {modulus_bits}-bit)"
    )
    ax.legend(loc="best")
    ax.grid(True, which="both")

    return _save(fig, output_dir, "montgomery_reductions.png")


# ---------------------------------------------------------------------------
# Plot 7: Memory Usage
# ---------------------------------------------------------------------------

def plot_memory_usage(
    df: pd.DataFrame, output_dir: str, exponent_bits: Optional[int] = None
) -> str:
    """
    Plot 7: Memory Usage vs Modulus Size.

    For a fixed exponent bit-length (default: the LARGEST tested), plots
    peak traced memory against modulus bit-length for every algorithm.
    Logarithmic y-axis, since sliding window's precomputed odd-power
    table (2^(window_size-1) entries, each roughly modulus_bits/8 bytes)
    can be meaningfully larger than the O(1)-auxiliary iterative
    algorithms, especially at the largest window sizes and moduli --
    exactly the memory/multiplication-count trade-off the project asks
    to characterize.
    """
    if exponent_bits is None:
        exponent_bits = int(df["exponent_bits"].max())

    subset = df[df["exponent_bits"] == exponent_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, algo in enumerate(_algorithm_order(subset["algorithm"].unique())):
        algo_df = subset[subset["algorithm"] == algo].sort_values("modulus_bits")
        if algo_df.empty:
            continue
        ax.plot(
            algo_df["modulus_bits"],
            algo_df["peak_memory_bytes"],
            label=algo,
            **_style_for(algo, i),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Modulus bit-length")
    ax.set_ylabel("Peak traced memory, bytes (log scale)")
    ax.set_title(f"Memory Usage vs. Modulus Size  (exponent $\\approx 2^{{{exponent_bits}}}$)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "memory.png")


# ---------------------------------------------------------------------------
# Plot 8: Runtime Comparison (all algorithms on one graph, bar form)
# ---------------------------------------------------------------------------

def plot_runtime_comparison_bar(
    df: pd.DataFrame, output_dir: str, modulus_bits: Optional[int] = None
) -> str:
    """
    Plot 8: Runtime Comparison -- all algorithms, grouped bar chart.

    Deliberately a DIFFERENT chart type from Plot 1 (bars instead of
    lines), snapshotting mean runtime at three representative exponent
    sizes (smallest, a middle value, and the largest tested) for a fixed
    modulus size. This "at a glance" comparison complements Plot 1's
    continuous trend view. Logarithmic y-axis, since naive's runtime at
    the largest shown exponent size can be orders of magnitude above
    every other algorithm's.
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    available_bits = sorted(subset["exponent_bits"].unique())
    if not available_bits:
        raise ValueError(f"No data available for modulus_bits={modulus_bits}")

    # Pick three representative exponent sizes: smallest, middle, largest.
    if len(available_bits) >= 3:
        chosen_bits = [available_bits[0], available_bits[len(available_bits) // 2], available_bits[-1]]
    else:
        chosen_bits = available_bits

    algorithms = _algorithm_order(subset["algorithm"].unique())
    n_algos = len(algorithms)
    n_groups = len(chosen_bits)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bar_width = 0.8 / n_algos
    group_positions = np.arange(n_groups)

    for i, algo in enumerate(algorithms):
        heights = []
        for bits in chosen_bits:
            row = subset[(subset["algorithm"] == algo) & (subset["exponent_bits"] == bits)]
            heights.append(float(row["mean_time_s"].iloc[0]) if not row.empty else np.nan)

        offsets = group_positions + (i - (n_algos - 1) / 2) * bar_width
        style = _style_for(algo, i)
        ax.bar(offsets, heights, width=bar_width, label=algo, color=style["color"])

    ax.set_yscale("log")
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"$2^{{{b}}}$" for b in chosen_bits])
    ax.set_xlabel("Exponent size")
    ax.set_ylabel("Mean runtime, seconds (log scale)")
    ax.set_title(f"Runtime Comparison Across All Algorithms  (modulus = {modulus_bits}-bit)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both", axis="y")

    return _save(fig, output_dir, "runtime_comparison.png")


# ---------------------------------------------------------------------------
# Plot 9: Speedup Relative to Naive
# ---------------------------------------------------------------------------

def plot_speedup_vs_naive(
    df: pd.DataFrame, output_dir: str, modulus_bits: Optional[int] = None
) -> str:
    """
    Plot 9: Speedup Relative to Naive vs Exponent Size.

    speedup(algorithm, bits) = mean_time(naive, bits) / mean_time(algorithm, bits)

    Restricted to the exponent sizes where naive was actually run (see
    benchmark.py's naive cutoff). Logarithmic y-axis is essentially
    mandatory here: speedups grow explosively (naive's O(2^bits) against
    everyone else's O(bits) means the ratio ITSELF grows exponentially
    in bits), easily spanning many orders of magnitude within the tested
    range.
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    naive_df = subset[subset["algorithm"] == "naive"].set_index("exponent_bits")["mean_time_s"]

    if naive_df.empty:
        raise ValueError(
            f"No naive results found at modulus_bits={modulus_bits}; cannot compute speedup."
        )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    other_algorithms = [a for a in _algorithm_order(subset["algorithm"].unique()) if a != "naive"]
    for i, algo in enumerate(other_algorithms):
        algo_df = subset[subset["algorithm"] == algo].set_index("exponent_bits")["mean_time_s"]
        common_bits = sorted(set(algo_df.index) & set(naive_df.index))
        if not common_bits:
            continue
        speedups = [naive_df[b] / algo_df[b] for b in common_bits]
        ax.plot(common_bits, speedups, label=algo, **_style_for(algo, i + 1))

    ax.set_yscale("log")
    ax.set_xlabel("Exponent bit-length")
    ax.set_ylabel("Speedup relative to naive, $\\times$ (log scale)")
    ax.set_title(f"Speedup Relative to Naive  (modulus = {modulus_bits}-bit)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "speedup_vs_naive.png")


# ---------------------------------------------------------------------------
# Plot 10: Theoretical Complexity vs Experimental Runtime
# ---------------------------------------------------------------------------

def plot_theoretical_vs_experimental(
    df: pd.DataFrame, output_dir: str, modulus_bits: Optional[int] = None
) -> str:
    """
    Plot 10: Theoretical Complexity vs Experimental Runtime.

    For each algorithm, overlays its MEASURED mean runtime (markers)
    against a THEORETICAL reference curve (dashed line) shaped according
    to its expected time complexity:

        * naive             : theoretical shape f(bits) = 2^bits   (Theta(E))
        * every other algo  : theoretical shape f(bits) = bits     (Theta(log E))

    Each theoretical curve is scaled by a single constant c, fitted so
    that c * f(bits) exactly matches the algorithm's OWN measured
    runtime at the smallest available exponent size (a simple,
    transparent one-point calibration -- no curve-fitting library
    required, and easy to audit by hand). The overlay then shows, for
    every larger exponent size, how closely the rest of the measured
    curve tracks the predicted shape: close tracking is direct
    experimental confirmation that the implementation's real-world
    growth rate matches its Big-O analysis; systematic drift would flag
    a mismatch worth investigating.
    """
    if modulus_bits is None:
        modulus_bits = int(df["modulus_bits"].max())

    subset = df[df["modulus_bits"] == modulus_bits]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for i, algo in enumerate(_algorithm_order(subset["algorithm"].unique())):
        algo_df = subset[subset["algorithm"] == algo].sort_values("exponent_bits")
        if algo_df.empty:
            continue

        bits = algo_df["exponent_bits"].to_numpy(dtype=float)
        measured = algo_df["mean_time_s"].to_numpy(dtype=float)

        if algo == "naive":
            shape = np.power(2.0, bits)
        else:
            shape = bits.copy()

        # One-point calibration: match theoretical curve to the FIRST
        # measured point for this algorithm.
        scale = measured[0] / shape[0] if shape[0] != 0 else 0.0
        theoretical = scale * shape

        style = _style_for(algo, i)
        ax.plot(bits, measured, label=f"{algo} (measured)", linestyle="none",
                marker=style["marker"], color=style["color"], markersize=7)
        ax.plot(bits, theoretical, label=f"{algo} (theoretical)", linestyle=":",
                color=style["color"], linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("Exponent bit-length")
    ax.set_ylabel("Runtime, seconds (log scale)")
    ax.set_title(f"Theoretical Complexity vs. Experimental Runtime  (modulus = {modulus_bits}-bit)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, which="both")

    return _save(fig, output_dir, "complexity.png")


# ---------------------------------------------------------------------------
# Driver: generate all plots at once
# ---------------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Generate all ten required plots and save them as high-resolution
    PNGs in output_dir.

    Returns
    -------
    Dict[str, str]
        Mapping from a short, descriptive plot key to the saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    return {
        "runtime_vs_exponent": plot_runtime_vs_exponent_size(df, output_dir),
        "runtime_vs_modulus": plot_runtime_vs_modulus_size(df, output_dir),
        "multiplications": plot_multiplication_counts(df, output_dir),
        "squarings": plot_squaring_counts(df, output_dir),
        "modulo_operations": plot_modulo_operation_counts(df, output_dir),
        "montgomery_reductions": plot_montgomery_reductions(df, output_dir),
        "memory_usage": plot_memory_usage(df, output_dir),
        "runtime_comparison": plot_runtime_comparison_bar(df, output_dir),
        "speedup_vs_naive": plot_speedup_vs_naive(df, output_dir),
        "theoretical_vs_experimental": plot_theoretical_vs_experimental(df, output_dir),
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all required plots from a benchmark results CSV."
    )
    parser.add_argument("--csv", type=str, default="results/benchmark.csv",
                         help="Path to the benchmark results CSV.")
    parser.add_argument("--output-dir", type=str, default="results",
                         help="Directory to write PNG plots into.")
    args = parser.parse_args()

    df = load_results(args.csv)
    print(f"Loaded {len(df)} benchmark records from {args.csv}")

    paths = generate_all_plots(df, args.output_dir)
    print(f"\nGenerated {len(paths)} plots:")
    for name, path in paths.items():
        print(f"  {name:30s} -> {path}")


if __name__ == "__main__":
    main()
