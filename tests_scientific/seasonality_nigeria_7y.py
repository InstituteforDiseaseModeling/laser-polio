import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import laser_polio as lp

###################################
######### USER PARAMETERS #########

# Sweep parameters
r0s = [5, 10, 15]
seasonal_amplitudes = [0.1, 0.5, 0.9]
seasonal_peak_doys = [120, 165, 210]
n_reps = 2

# Base parameters
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
r0 = 10
migration_method = "radiation"
node_seeding_dispersion = 1.0
max_migr_frac = 0.1
vx_prob_ri = 0.0
use_pim_scalars = False
results_path = "results/tests_scientific/seasonality_jigawa_zamfara_niger_7y"


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

print(f"Running seasonality sweep with {len(r0s)} r0s x {len(seasonal_amplitudes)} amplitudes x {len(seasonal_peak_doys)} peak days")

# Run sweep
for r0 in r0s:
    for seasonal_amplitude in seasonal_amplitudes:
        for seasonal_peak_doy in seasonal_peak_doys:
            key = (r0, seasonal_amplitude, seasonal_peak_doy)

            print(f"\nRunning R0={r0}, amplitude={seasonal_amplitude}, peak_doy={seasonal_peak_doy}")

            for rep in range(n_reps):
                print(f"  ↳ Rep {rep + 1}/{n_reps}")

                if key == (r0s[0], seasonal_amplitudes[0], seasonal_peak_doys[0]) and rep == 0:
                    save_plots = True
                    results_path_sim = results_path + f"/sim_r0_{r0}_amp_{seasonal_amplitude}_peak_{seasonal_peak_doy}_rep_{rep}"
                else:
                    save_plots = False
                    results_path_sim = results_path

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
                    max_migr_frac=max_migr_frac,
                    vx_prob_ri=vx_prob_ri,
                    use_pim_scalars=use_pim_scalars,
                    seasonal_amplitude=seasonal_amplitude,
                    seasonal_peak_doy=seasonal_peak_doy,
                    results_path=results_path + f"/sim_r0_{r0}_amp_{seasonal_amplitude}_peak_{seasonal_peak_doy}_rep_{rep}",
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
                key_rep = (r0, seasonal_amplitude, seasonal_peak_doy, rep)
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


def _plot_metric_for_r0(
    r0,
    metric_name,  # "infected" or "npp"
    per_rep_dict,  # e.g., infected_timeseries or npp_timeseries
    avg_dict,  # e.g., infected_timeseries_average or npp_timeseries_average
    seasonal_amplitudes,
    seasonal_peak_doys,
    n_reps,
    n_days,
    results_path,
    ylabel,
    filename_prefix,
):
    nrows, ncols = len(seasonal_amplitudes), len(seasonal_peak_doys)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(f"{metric_name.upper()} over time (sum over nodes) — Seasonality Sweep — R0={r0}", fontsize=16)

    # Ensure save dir
    os.makedirs(results_path, exist_ok=True)

    first_rep_plotted = False  # for legend labeling
    first_mean_plotted = False

    for i, seasonal_amplitude in enumerate(seasonal_amplitudes):
        for j, seasonal_peak_doy in enumerate(seasonal_peak_doys):
            ax = axes[i, j]

            # 1) Plot each replicate (light lines)
            rep_series = []
            for rep in range(n_reps):
                k = (r0, seasonal_amplitude, seasonal_peak_doy, rep)
                ts = per_rep_dict.get(k)
                if ts is None:
                    continue
                ts = np.asarray(ts)
                rep_series.append(ts)
                ax.plot(ts, alpha=0.30, linewidth=1.0, label="Replicate" if not first_rep_plotted else None)
                first_rep_plotted = True

            # 2) Plot mean across reps (thick line)
            base_key = (r0, seasonal_amplitude, seasonal_peak_doy)
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

            ax.set_title(f"Amp={seasonal_amplitude}, Peak DoY={seasonal_peak_doy}")
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
    outfile = os.path.join(results_path, f"{filename_prefix}_r0_{r0}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


# =======================
# Call for each R0 & metric
# =======================
for r0 in r0s:
    # Infected
    _plot_metric_for_r0(
        r0=r0,
        metric_name="infected",
        per_rep_dict=infected_timeseries,  # keys: (r0, amp, peak, rep)
        avg_dict=infected_timeseries_average,  # keys: (r0, amp, peak)
        seasonal_amplitudes=seasonal_amplitudes,
        seasonal_peak_doys=seasonal_peak_doys,
        n_reps=n_reps,
        n_days=n_days,
        results_path=results_path,
        ylabel="Infected (sum over nodes)",
        filename_prefix="infected_time_series",
    )

    # NPP
    _plot_metric_for_r0(
        r0=r0,
        metric_name="npp",
        per_rep_dict=npp_timeseries,  # keys: (r0, amp, peak, rep)
        avg_dict=npp_timeseries_average,  # keys: (r0, amp, peak)
        seasonal_amplitudes=seasonal_amplitudes,
        seasonal_peak_doys=seasonal_peak_doys,
        n_reps=n_reps,
        n_days=n_days,
        results_path=results_path,
        ylabel="New Potentially Paralyzed (sum over nodes)",
        filename_prefix="npp_time_series",
    )

print("Done!")
