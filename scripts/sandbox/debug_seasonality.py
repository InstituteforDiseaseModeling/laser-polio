import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_path = "results/debug_seasonality"
os.makedirs(results_path, exist_ok=True)


# Generate dates for the year 2018
dates = pd.date_range(start="2018-01-01", end="2018-12-31")
doys = [d.timetuple().tm_yday for d in dates]

# Parameters
seasonal_amplitude = 0.4
peak_days = [90, 180, 210, 300]  # example DOYs

# Setup figure with extra space for a legend subplot
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.2])
axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
legend_ax = fig.add_subplot(gs[:, 2])  # full-height subplot for legend
legend_ax.axis("off")  # hide axis for clean legend

# Placeholder for legend handles
legend_handles = []

for i, peak_doy in enumerate(peak_days):
    seasonality_current = []
    seasonality_suggested = []

    for doy in doys:
        seasonality_current.append(1 + seasonal_amplitude * np.cos((2 * np.pi * doy / 366) + (2 * np.pi * peak_doy / 366)))
        seasonality_suggested.append(1 + seasonal_amplitude * np.cos(2 * np.pi * (doy - peak_doy) / 366))

    ax = axs[i]
    (line1,) = ax.plot(doys, seasonality_current, label="Current")
    (line2,) = ax.plot(doys, seasonality_suggested, label="Suggested")
    line3 = ax.axvline(x=peak_doy, color="r", linestyle="--", label="Peak DOY")

    if i == 0:
        legend_handles = [line1, line2, line3]

    ax.set_title(f"Peak DOY = {peak_doy}")
    ax.set_ylim([1 - seasonal_amplitude * 1.1, 1 + seasonal_amplitude * 1.1])
    ax.grid(True)

# Add a clean legend in the side subplot
legend_ax.legend(handles=legend_handles, labels=[h.get_label() for h in legend_handles], loc="center", frameon=False)

# Title and layout
fig.suptitle("Seasonal Infectivity Modifiers with Varying Peak DOY", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.95, 0.96])
plt.savefig(os.path.join(results_path, "plot_seasonality_bug.png"))
plt.close()
