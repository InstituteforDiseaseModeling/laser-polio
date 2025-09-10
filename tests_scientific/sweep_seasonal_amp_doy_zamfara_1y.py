import os

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

# Base parameters from demo_nigeria.py - modified for faster execution
regions = ["ZAMFARA"]  # Use smaller region instead of full Nigeria
start_year = 2018
n_days = 365  # 1 year instead of 3 for faster execution
pop_scale = 1
init_region = "ANKA"  # Use region within ZAMFARA
init_prev = 200  # Reduce initial prevalence
r0 = 10
migration_method = "radiation"
radiation_k_log10 = -0.3
max_migr_frac = 0.1
vx_prob_ri = 0.0
missed_frac = 0.1
use_pim_scalars = False
results_path = "results/tests_scientific/seasonality_zamfara_1y"

# No seed schedule for simplicity - just use init_prev
seed_schedule = None

# Seasonality sweep parameters
seasonal_amplitudes = [0.1, 0.5, 0.9]
seasonal_peak_doys = [90, 180, 270]  # Spring, Summer, Fall peaks
n_reps = 10  # Single rep for faster execution

######### END OF USER PARS ########
###################################

# Create result storage
infected_timeseries = {}
total_infected = {}
infected_timeseries_average = {}

print(f"Running seasonality sweep with {len(seasonal_amplitudes)} amplitudes x {len(seasonal_peak_doys)} peak days")

# Run sweep
for seasonal_amplitude in seasonal_amplitudes:
    for seasonal_peak_doy in seasonal_peak_doys:
        ts_list = []  # collect reps for this (amplitude, peak_doy)

        print(f"\nRunning amplitude={seasonal_amplitude}, peak_doy={seasonal_peak_doy}")

        for rep in range(n_reps):
            print(f"  ↳ Rep {rep + 1}/{n_reps}")

            sim = lp.run_sim(
                regions=regions,
                start_year=start_year,
                n_days=n_days,
                pop_scale=pop_scale,
                init_region=init_region,
                init_prev=init_prev,
                seed_schedule=seed_schedule,
                r0=r0,
                migration_method=migration_method,
                radiation_k_log10=radiation_k_log10,
                max_migr_frac=max_migr_frac,
                vx_prob_ri=vx_prob_ri,
                missed_frac=missed_frac,
                use_pim_scalars=use_pim_scalars,
                seasonal_amplitude=seasonal_amplitude,
                seasonal_peak_doy=seasonal_peak_doy,
                results_path=results_path,
                save_plots=False,
                save_data=False,
                verbose=0,
                seed=rep,
                run=True,
            )

            # Store timeseries
            ts = np.sum(sim.results.I_by_strain[:, :, 0], axis=1)
            key_rep = (seasonal_amplitude, seasonal_peak_doy, rep)
            infected_timeseries[key_rep] = ts
            total_infected[key_rep] = ts.sum()
            ts_list.append(ts)

        # stack and compute aggregates for this (amp, peak)
        arr = np.stack(ts_list, axis=0)  # shape: (n_reps, T, ...)
        infected_timeseries_average[(seasonal_amplitude, seasonal_peak_doy)] = arr.mean(axis=0)

os.makedirs(results_path, exist_ok=True)

# Create plots
print("\nCreating plots...")


# --- Plot: per-rep timeseries + mean across reps for each (amp, peak) ---
nrows, ncols = len(seasonal_amplitudes), len(seasonal_peak_doys)
fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows), sharex=True, sharey=True)
axes = np.atleast_2d(axes)  # ensures 2D indexing even if nrows/ncols == 1
fig.suptitle("Infected Over Time (summed over nodes) - Seasonality Sweep", fontsize=16)

for i, seasonal_amplitude in enumerate(seasonal_amplitudes):
    for j, seasonal_peak_doy in enumerate(seasonal_peak_doys):
        ax = axes[i, j]
        combo_key = (seasonal_amplitude, seasonal_peak_doy)

        # 1) Plot each replicate (light lines)
        rep_series = []
        for rep in range(n_reps):
            k = (seasonal_amplitude, seasonal_peak_doy, rep)
            ts = infected_timeseries.get(k)
            if ts is None:
                continue
            ts = np.asarray(ts)
            rep_series.append(ts)
            ax.plot(ts, alpha=0.30, linewidth=1.0, label="Replicate" if (i == 0 and j == 0 and rep == 0) else None)

        # 2) Plot mean across reps (thick line)
        mean_ts = infected_timeseries_average[combo_key]
        mean_ts = np.asarray(mean_ts)
        ax.plot(mean_ts, linewidth=2.5, label="Mean across reps" if (i == 0 and j == 0) else None)

        # # 3) Optional: 95% CI band (recompute std from stored per-rep series)
        # if len(rep_series) >= 2:
        #     arr = np.stack(rep_series, axis=0)  # (n_reps, time)
        #     std_ts = arr.std(axis=0, ddof=1)
        #     sem_ts = std_ts / np.sqrt(arr.shape[0])
        #     t = np.arange(mean_ts.shape[0])
        #     ax.fill_between(
        #         t,
        #         mean_ts - 1.96 * sem_ts,
        #         mean_ts + 1.96 * sem_ts,
        #         alpha=0.20,
        #         linewidth=0,
        #         label="95% CI" if (i == 0 and j == 0) else None,
        #     )

        ax.set_title(f"Amp={seasonal_amplitude}, Peak DoY={seasonal_peak_doy}")
        ax.grid(True, alpha=0.3)
        if i == nrows - 1:
            ax.set_xlabel("Time (days)")
        if j == 0:
            ax.set_ylabel("Infected (sum over nodes)")

# Single legend to avoid clutter
handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper right", frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(results_path, exist_ok=True)
plt.savefig(f"{results_path}/infected_time_series.png", dpi=300, bbox_inches="tight")
plt.close()


# # Plot 1: Time series of infections for each parameter combination + average
# fig, axes = plt.subplots(len(seasonal_amplitudes), len(seasonal_peak_doys), figsize=(15, 12), sharex=True, sharey=True)
# fig.suptitle("Infected Over Time - Seasonality Sweep", fontsize=16)

# for i, seasonal_amplitude in enumerate(seasonal_amplitudes):
#     for j, seasonal_peak_doy in enumerate(seasonal_peak_doys):
#         ax = axes[i, j]
#         key = (seasonal_amplitude, seasonal_peak_doy)
#         I_data = infected_results[key]

#         # Plot total infections over time (sum across all nodes)
#         total_I = I_data.sum(axis=1)
#         ax.plot(total_I, label="Total Infected", linewidth=2)

#         # Add some individual node traces if there are multiple nodes
#         if I_data.shape[1] > 1:
#             # Plot top few nodes with most total infections
#             node_totals = I_data.sum(axis=0)
#             top_nodes = np.argsort(node_totals)[-3:]  # Top 3 nodes
#             for node_idx in top_nodes:
#                 if node_totals[node_idx] > 0:  # Only plot if there were infections
#                     ax.plot(I_data[:, node_idx], alpha=0.7, linestyle="--", label=f"Node {node_idx}")

#         ax.set_title(f"Amp={seasonal_amplitude}, Peak DoY={seasonal_peak_doy}")
#         ax.grid(True, alpha=0.3)
#         if i == len(seasonal_amplitudes) - 1:
#             ax.set_xlabel("Time (days)")
#         if j == 0:
#             ax.set_ylabel("Infected")

#         # Add legend only for first subplot to avoid clutter
#         if i == 0 and j == 0:
#             ax.legend(fontsize="small")

# plt.tight_layout()
# plt.savefig(f"{results_path}/infected_time_series.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Plot 2: Heatmap of total infections
# fig, ax = plt.subplots(figsize=(10, 8))
# im = ax.imshow(total_infected_matrix, cmap="viridis", aspect="auto")
# ax.set_xticks(range(len(seasonal_peak_doys)))
# ax.set_yticks(range(len(seasonal_amplitudes)))
# ax.set_xticklabels([f"DoY {doy}" for doy in seasonal_peak_doys])
# ax.set_yticklabels([f"Amp {amp}" for amp in seasonal_amplitudes])
# ax.set_xlabel("Seasonal Peak Day of Year")
# ax.set_ylabel("Seasonal Amplitude")
# ax.set_title("Total Infected Cases - Seasonality Parameter Sweep")

# # Add text annotations
# for i in range(len(seasonal_amplitudes)):
#     for j in range(len(seasonal_peak_doys)):
#         text = ax.text(j, i, f"{total_infected_matrix[i, j]:.0f}", ha="center", va="center", color="white", fontweight="bold")

# plt.colorbar(im, ax=ax, label="Total Infected")
# plt.tight_layout()
# plt.savefig(f"{results_path}/total_infected_heatmap.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Plot 3: Line plot showing effect of seasonal amplitude for each peak day
# fig, ax = plt.subplots(figsize=(10, 6))
# for j, seasonal_peak_doy in enumerate(seasonal_peak_doys):
#     values = total_infected_matrix[:, j]
#     ax.plot(seasonal_amplitudes, values, marker="o", linewidth=2, label=f"Peak DoY {seasonal_peak_doy}")

# ax.set_xlabel("Seasonal Amplitude")
# ax.set_ylabel("Total Infected")
# ax.set_title("Effect of Seasonal Amplitude on Total Infections")
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f"{results_path}/amplitude_effect.png", dpi=300, bbox_inches="tight")
# plt.close()

# # Plot 4: Line plot showing effect of seasonal peak timing for each amplitude
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, seasonal_amplitude in enumerate(seasonal_amplitudes):
#     values = total_infected_matrix[i, :]
#     ax.plot(seasonal_peak_doys, values, marker="s", linewidth=2, label=f"Amplitude {seasonal_amplitude}")

# ax.set_xlabel("Seasonal Peak Day of Year")
# ax.set_ylabel("Total Infected")
# ax.set_title("Effect of Seasonal Timing on Total Infections")
# ax.legend()
# ax.grid(True, alpha=0.3)
# # Add season labels
# season_labels = ["Winter→Spring", "Summer", "Fall"]
# for i, (doy, label) in enumerate(zip(seasonal_peak_doys, season_labels, strict=False)):
#     ax.text(doy, ax.get_ylim()[1] * 0.9, label, ha="center", fontsize=9, alpha=0.7)
# plt.tight_layout()
# plt.savefig(f"{results_path}/timing_effect.png", dpi=300, bbox_inches="tight")
# plt.close()

# print("\nSeasonality sweep complete!")
# print(f"Results saved to: {results_path}")
# print("Total infected cases matrix:")
# print(f"Rows = Seasonal Amplitudes: {seasonal_amplitudes}")
# print(f"Cols = Seasonal Peak DoYs: {seasonal_peak_doys}")
# print(total_infected_matrix)

sc.printcyan("Done.")
