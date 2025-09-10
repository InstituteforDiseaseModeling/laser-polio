import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc

import laser_polio as lp

"""
This tests the impact of random seed on the outbreak size.
Ideally, the outbreak size should be different for different seeds.
"""


###################################
######### USER PARAMETERS #########

n_reps = 100
results_path = "results/tests_scientific/rand_seed"

regions = ["ZAMFARA"]
start_year = 2018
n_days = 365
pop_scale = 1 / 1
init_region = "ANKA"
init_prev = 200
r0 = 10
migration_method = "radiation"
radiation_k_log10 = -0.3
max_migr_frac = 0.1
verbose = 0
vx_prob_ri = 0.0
missed_frac = 0.1
seed_schedule = [
    {"date": "2018-01-02", "dot_name": "AFRO:NIGERIA:ZAMFARA:BAKURA", "prevalence": 200},  # day 1
    {"date": "2018-11-07", "dot_name": "AFRO:NIGERIA:ZAMFARA:GUMMI", "prevalence": 200},  # day 2
]
save_plots = False
save_data = False
plot_pars = False
save_init_pop = False
init_pop_file = None


######### END OF USER PARS ########
###################################

os.makedirs(results_path, exist_ok=True)


# ----- Use Laser-Polio to simulate the outbreak size -----#

I_series_store = {}  # Key: (heterogeneity, r0), Value: 1D array of I over time
new_exposed_store = {}  # Key: (heterogeneity, r0), Value: 1D array of new exposed over time
records = []
seeds = range(n_reps)
for seed in seeds:
    print(f"\nSeed = {seed}")

    sim = lp.run_sim(
        seed=seed,
        results_path=results_path,
        regions=regions,
        start_year=start_year,
        n_days=n_days,
        pop_scale=pop_scale,
        init_region=init_region,
        init_prev=init_prev,
        save_plots=save_plots,
        save_data=save_data,
        plot_pars=plot_pars,
        verbose=verbose,
        r0=r0,
        migration_method=migration_method,
        radiation_k_log10=radiation_k_log10,
        max_migr_frac=max_migr_frac,
        save_init_pop=save_init_pop,
        vx_prob_ri=vx_prob_ri,
        init_pop_file=init_pop_file,
        # seed_schedule=seed_schedule,
        missed_frac=missed_frac,
        use_pim_scalars=True,
    )

    # Save the results
    I_series_store[(seed)] = np.sum(sim.results.I_by_strain[:, :, 0], axis=1)  # all times, single node, VDPV2 strain
    new_exposed = np.sum(sim.results.new_exposed_by_strain[:, :, 0], axis=1)  # all times, single node, VDPV2 strain
    new_exposed_store[(seed)] = new_exposed
    records.append(
        {
            "seed": seed,
            "init_prev": init_prev,
            "total_new_exposed": new_exposed.sum(),
        }
    )


# Convert records to DataFrame
df = pd.DataFrame.from_records(records)
df["total_new_exposed"] = df["total_new_exposed"]

# ----- Plotting -----#


# Plot the Infection time series
plt.figure(figsize=(10, 6))
for seed in seeds:
    I_t = I_series_store[seed]
    plt.plot(I_t, label=seed)
plt.title("Infected Over Time")
plt.xlabel("Time (Timesteps)")
plt.ylabel("Infected Individuals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "infection_time_series.png"))
# plt.show()
plt.close()


# Plot the histogram of total new exposed
plt.figure(figsize=(8, 6))
plt.hist(df["total_new_exposed"], bins=30, edgecolor="black")
plt.title("Histogram of total_new_exposed")
plt.xlabel("Total New Exposed")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "total_new_exposed_histogram.png"))
# plt.show()
plt.close()


sc.printcyan("Done.")
