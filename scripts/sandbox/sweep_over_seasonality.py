import os
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np

from laser_polio.utils import get_seasonality

results_path = "results/sweep_over_seasonality"


class MockSim:
    def __init__(self, seasonal_amplitude, seasonal_peak_doy):
        self.pars = {"seasonal_amplitude": seasonal_amplitude, "seasonal_peak_doy": seasonal_peak_doy}
        self.t = 0
        # Create a full year of dates starting from Jan 1, 2024
        start_date = datetime(2024, 1, 1)
        self.datevec = [start_date + timedelta(days=x) for x in range(365)]


def plot_seasonality_sweep(results_path=results_path):
    # Create parameter ranges
    amplitudes = np.linspace(0, 0.4, 5)  # 5 values from 0 to 1
    peak_doys = np.linspace(120, 300, 5)  # 5 values spread across the year

    # Create figure with subplots
    fig, axes = plt.subplots(len(amplitudes), 1, figsize=(12, 15))
    fig.suptitle("Seasonality Patterns for Different Parameter Values", fontsize=16)

    # Create x-axis dates for plotting
    dates = [datetime(2024, 1, 1) + timedelta(days=x) for x in range(365)]

    # Calculate overall y-axis limits
    all_seasonality = []
    for amplitude in amplitudes:
        for peak_doy in peak_doys:
            sim = MockSim(amplitude, peak_doy)
            seasonality = []
            for day in range(365):
                sim.t = day
                seasonality.append(get_seasonality(sim))
            all_seasonality.extend(seasonality)

    y_min = min(all_seasonality)
    y_max = max(all_seasonality)

    # Plot each combination
    for i, amplitude in enumerate(amplitudes):
        ax = axes[i]
        for peak_doy in peak_doys:
            # Create mock sim object with parameters
            sim = MockSim(amplitude, peak_doy)

            # Calculate seasonality for each day
            seasonality = []
            for day in range(365):
                sim.t = day
                seasonality.append(get_seasonality(sim))

            # Plot the seasonality curve
            label = f"Peak DoY: {int(peak_doy)}"
            ax.plot(dates, seasonality, label=label)

        ax.set_title(f"Seasonal Amplitude: {amplitude:.2f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Seasonality Factor")
        ax.grid(True)
        ax.legend()

        # Set consistent axis limits
        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, "seasonality_sweep.png"))
    plt.close()


if __name__ == "__main__":
    plot_seasonality_sweep()
