from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import laser_polio as lp


def analyze_r0_scalars(
    slopes=None,
    intercepts=None,
    data_path="data/compiled_cbr_pop_ri_sia_underwt_africa.csv",
    save_plot=True,
    output_dir="results/sweep_over_r0_scalar_pars",
):
    """
    Analyze how different r0_scalar_wt_slope and r0_scalar_wt_intercept values
    affect the R0 scaling based on underweight proportions.

    Args:
        slopes (list): List of slope values to analyze
        intercepts (list): List of intercept values to analyze
        data_path (str): Path to the data file containing prop_underwt values
        save_plot (bool): Whether to save the plots
        output_dir (str): Directory to save plots
    """
    slopes = np.linspace(0, 100, 10) if slopes is None else slopes
    intercepts = np.linspace(0, 1, 10) if intercepts is None else intercepts

    # Load data and get prop_underwt values
    df = pd.read_csv(lp.root / data_path)
    underwt = df["prop_underwt"].values

    # Create range of x values for smooth curves
    x = np.linspace(underwt.min(), underwt.max(), 100)

    # Create figure for slopes
    plt.figure(figsize=(12, 8))
    base_intercept = 0.2  # Use default intercept
    for slope in slopes:
        y = 1 / (1 + np.exp(slope * (0.22 - x))) + base_intercept
        plt.plot(x, y, label=f"Slope = {slope}")

    plt.xlabel("Proportion Underweight")
    plt.ylabel("R0 Scalar")
    plt.title(f"R0 Scalar vs Underweight Proportion\n(Fixed intercept = {base_intercept})")
    plt.legend()
    plt.grid(True)

    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "r0_scalar_slopes.png")

    # Create figure for intercepts
    plt.figure(figsize=(12, 8))
    base_slope = 24  # Use default slope
    for intercept in intercepts:
        y = 1 / (1 + np.exp(base_slope * (0.22 - x))) + intercept
        plt.plot(x, y, label=f"Intercept = {intercept}")

    plt.xlabel("Proportion Underweight")
    plt.ylabel("R0 Scalar")
    plt.title(f"R0 Scalar vs Underweight Proportion\n(Fixed slope = {base_slope})")
    plt.legend()
    plt.grid(True)

    if save_plot:
        plt.savefig(output_path / "r0_scalar_intercepts.png")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Underweight Proportion Range: {underwt.min():.3f} to {underwt.max():.3f}")
    print(f"Mean Underweight Proportion: {underwt.mean():.3f}")
    print(f"Median Underweight Proportion: {np.median(underwt):.3f}")


if __name__ == "__main__":
    analyze_r0_scalars()
