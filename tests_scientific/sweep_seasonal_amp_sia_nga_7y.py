import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import laser_polio as lp

###################################
######### USER PARAMETERS #########

# Sweep parameters
seasonal_amplitudes = [0.0, 0.1, 0.2]
sia_re_centers = [1e-10, 0.5]  # 1e-10: no SIA, 0.5: medium SIA
n_reps = 2

# Base parameters
r0 = 14
seasonal_peak_doy = 159
regions = ["NIGERIA:JIGAWA", "NIGERIA:ZAMFARA", "NIGERIA:NIGER"]
start_year = 2017
n_days = 365 * 2
pop_scale = 1
init_region = "BIRINIWA"
init_prev = 0
seed_schedule = [
    {"date": "2017-10-01", "dot_name": "AFRO:NIGERIA:JIGAWA:HADEJIA", "prevalence": 100},
    {"date": "2017-10-01", "dot_name": "AFRO:NIGERIA:JIGAWA:GARKI", "prevalence": 100},
    {"date": "2020-07-01", "dot_name": "AFRO:NIGERIA:ZAMFARA:TALATA_MAFARA", "prevalence": 100},
    {"date": "2020-10-01", "dot_name": "AFRO:NIGERIA:NIGER:SULEJA", "prevalence": 100},
]
migration_method = "radiation"
node_seeding_dispersion = 1.0
max_migr_frac = 0.1
vx_prob_ri = 0.0
use_pim_scalars = False
results_path = "results/tests_scientific/sweep_seasonal_amp_sia_nga_7y"

######### END OF USER PARS ########
###################################

# Create result storage
infected_timeseries = {}
npp_timeseries = {}
infected_timeseries_average = {}
npp_timeseries_average = {}

infected_sum = defaultdict(lambda: np.zeros(n_days + 1, dtype=np.float64))
npp_sum = defaultdict(lambda: np.zeros(n_days + 1, dtype=np.float64))
rep_count = defaultdict(int)

infected_timeseries_average = {}
npp_timeseries_average = {}

print(f"Running sweep with {len(seasonal_amplitudes)} amplitudes x {len(sia_re_centers)} SIA centers")

# Run sweep
for seasonal_amplitude in seasonal_amplitudes:
    for sia_re_center in sia_re_centers:
        key = (seasonal_amplitude, sia_re_center)

        print(f"\nRunning amplitude={seasonal_amplitude}, SIA center={sia_re_center}")

        for rep in range(n_reps):
            print(f"  ↳ Rep {rep + 1}/{n_reps}")

            if key == (seasonal_amplitudes[0], sia_re_centers[0]) and rep == 0:
                save_plots = True
                results_path_sim = results_path + f"/sim_amp_{seasonal_amplitude}_sia_{sia_re_center}_rep_{rep}"
            else:
                save_plots = False
                results_path_sim = results_path

            sim = lp.run_sim(
                sia_re_center=sia_re_center,
                regions=regions,
                start_year=start_year,
                n_days=n_days,
                pop_scale=pop_scale,
                init_region=init_region,
                init_prev=init_prev,
                seed_schedule=seed_schedule,
                r0=r0,
                migration_method=migration_method,
                max_migr_frac=max_migr_frac,
                vx_prob_ri=vx_prob_ri,
                use_pim_scalars=use_pim_scalars,
                seasonal_amplitude=seasonal_amplitude,
                seasonal_peak_doy=seasonal_peak_doy,
                results_path=results_path + f"/sim_amp_{seasonal_amplitude}_sia_{sia_re_center}_rep_{rep}",
                save_plots=save_plots,
                save_data=False,
                verbose=0,
                seed=rep,
                run=True,
            )

            # Per-rep daily series
            inf_ts = np.sum(sim.results.I_by_strain[:, :, 0], axis=1)
            npp_ts = np.sum(sim.results.new_potentially_paralyzed, axis=1)

            # (Optional) keep per-rep if you still want them
            key_rep = (seasonal_amplitude, sia_re_center, rep)
            infected_timeseries[key_rep] = inf_ts
            npp_timeseries[key_rep] = npp_ts

            # Accumulate for average
            infected_sum[key] += inf_ts
            npp_sum[key] += npp_ts
            rep_count[key] += 1

        # finalize averages for this combo
        infected_timeseries_average[key] = infected_sum[key] / rep_count[key]
        npp_timeseries_average[key] = npp_sum[key] / rep_count[key]


os.makedirs(results_path, exist_ok=True)

# Create plots
print("\nCreating plots...")


def _plot_metric_sweep(
    metric_name,  # "infected" or "npp"
    per_rep_dict,  # e.g., infected_timeseries or npp_timeseries
    avg_dict,  # e.g., infected_timeseries_average or npp_timeseries_average
    seasonal_amplitudes,
    sia_re_centers,
    n_reps,
    n_days,
    results_path,
    ylabel,
    filename_prefix,
):
    nrows, ncols = len(seasonal_amplitudes), len(sia_re_centers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(f"{metric_name.upper()} over time (sum over nodes) — Seasonal Amplitude vs SIA Center Sweep", fontsize=16)

    # Ensure save dir
    os.makedirs(results_path, exist_ok=True)

    first_rep_plotted = False  # for legend labeling
    first_mean_plotted = False

    for i, seasonal_amplitude in enumerate(seasonal_amplitudes):
        for j, sia_re_center in enumerate(sia_re_centers):
            ax = axes[i, j]

            # 1) Plot each replicate (light lines)
            rep_series = []
            for rep in range(n_reps):
                k = (seasonal_amplitude, sia_re_center, rep)
                ts = per_rep_dict.get(k)
                if ts is None:
                    continue
                ts = np.asarray(ts)
                rep_series.append(ts)
                ax.plot(ts, alpha=0.30, linewidth=1.0, label="Replicate" if not first_rep_plotted else None)
                first_rep_plotted = True

            # 2) Plot mean across reps (thick line)
            base_key = (seasonal_amplitude, sia_re_center)
            mean_ts = avg_dict.get(base_key)

            # If the precomputed mean is missing, compute it from whatever reps we have
            if mean_ts is None:
                if rep_series:
                    mean_ts = np.mean(np.stack(rep_series, axis=0), axis=0)
                else:
                    # fallback to zeros if nothing is present
                    mean_ts = np.zeros(n_days, dtype=float)

            mean_ts = np.asarray(mean_ts)
            ax.plot(mean_ts, linewidth=2.5, label="Mean across reps" if not first_mean_plotted else None)
            first_mean_plotted = True

            sia_label = "No SIA" if sia_re_center == 1e-10 else f"SIA center={sia_re_center}"
            ax.set_title(f"Amp={seasonal_amplitude}, {sia_label}")
            ax.grid(True, alpha=0.3)
            if i == nrows - 1:
                ax.set_xlabel("Time (days)")
            if j == 0:
                ax.set_ylabel(ylabel)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    outfile = os.path.join(results_path, f"{filename_prefix}_sweep.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


# =======================
# Create plots for metrics
# =======================

# Infected
_plot_metric_sweep(
    metric_name="infected",
    per_rep_dict=infected_timeseries,  # keys: (amp, sia_center, rep)
    avg_dict=infected_timeseries_average,  # keys: (amp, sia_center)
    seasonal_amplitudes=seasonal_amplitudes,
    sia_re_centers=sia_re_centers,
    n_reps=n_reps,
    n_days=n_days,
    results_path=results_path,
    ylabel="Infected (sum over nodes)",
    filename_prefix="infected_time_series",
)

# NPP
_plot_metric_sweep(
    metric_name="npp",
    per_rep_dict=npp_timeseries,  # keys: (amp, sia_center, rep)
    avg_dict=npp_timeseries_average,  # keys: (amp, sia_center)
    seasonal_amplitudes=seasonal_amplitudes,
    sia_re_centers=sia_re_centers,
    n_reps=n_reps,
    n_days=n_days,
    results_path=results_path,
    ylabel="New Potentially Paralyzed (sum over nodes)",
    filename_prefix="npp_time_series",
)

print("Done!")
