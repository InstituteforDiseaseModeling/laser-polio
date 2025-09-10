import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
from scipy.optimize import fsolve

import laser_polio as lp

"""
This tests the outbreak size (total number of infections) across different R0 values. 
The goal is for our model to match expectations from a simple SIR model (i.e., the Kermack-McKendrick model). 

The key assumptions are: 
- Disease states are SIR (i.e., no exposed state), but the exposed state is core to our model so here we give it a duration of 0.
- A fully susceptible population
- No vital rates (births, deaths, etc.)
- No seasonal variation
- No vaccination
- No migration (i.e., a single node)
- No heterogeneity

It is based on the notebook: https://github.com/InstituteforDiseaseModeling/laser-generic/blob/main/notebooks/04_SIR_nobirths_outbreak_size.ipynb
"""


###################################
######### USER PARAMETERS #########

r0_values = np.linspace(1, 10, 15)  # Sweep over R0 values
heterogeneity_values = [True, False]  # Sweep over heterogeneity
regions = ["ZAMFARA:ANKA"]
n_days = 365 * 2
init_pop = 1e6
pop_scale = 1e6 / 251573  # To scale up to 1e6
init_region = "ANKA"
init_prev = 20 / init_pop
init_immun_scalar = 0.0  # ensure that entire population is susceptible
ipv_vx = False
r0_scalar_wt_slope = 0.0  # ensures that r0_scalars = 1.0
r0_scalar_wt_intercept = 0.5  # ensures that r0_scalars = 1.0
seasonal_amplitude = 0.0  # no seasonality
cbr = np.array([0])  # no births or deaths
vx_prob_ri = None  # no routine immunization
vx_prob_sia = None  # no SIA
n_reps = 1
results_path = "results/tests_scientific/outbreak_size"


######### END OF USER PARS ########
###################################

os.makedirs(results_path, exist_ok=True)


# ----- Calculate the expected final size using the Kermack-McKendrick model -----#


def KM_limit(z, r0, S0, I0):
    if r0 * S0 < 1:
        return 0
    else:
        return z - S0 * (1 - np.exp(-r0 * (z + I0)))


population = init_pop
inf_mean = 24
init_inf = 20
S0 = 1.0
S0s = [1.0]
output = pd.DataFrame(list(itertools.product(r0_values, S0s)), columns=["r0", "S0"])
output["I_inf_exp"] = [
    fsolve(KM_limit, 0.5 * (r0 * S0 >= 1), args=(r0, S0, init_inf / population))[0]
    for r0, S0 in zip(output["r0"], output["S0"], strict=False)
]
output["S_inf_exp"] = output["S0"] - output["I_inf_exp"]


# ----- Use Laser-Polio to simulate the outbreak size -----#

I_series_store = {}  # Key: (heterogeneity, r0), Value: 1D array of I over time
new_exposed_store = {}  # Key: (heterogeneity, r0), Value: 1D array of new exposed over time
records = []
for r0 in r0_values:
    print(f"\nSweeping r0 = {r0:.2f}")

    for heterogeneity in heterogeneity_values:
        label = f"{'Hetero' if heterogeneity else 'NoHetero'}"
        print(f" → Config: {label}")

        for rep in range(n_reps):
            sim = lp.run_sim(
                regions=regions,
                n_days=n_days,
                pop_scale=pop_scale,
                init_region=init_region,
                init_prev=init_prev,
                results_path=results_path,
                save_plots=False,
                save_data=False,
                init_pop=init_pop,
                r0=r0,
                seasonal_amplitude=seasonal_amplitude,
                cbr=cbr,
                vx_prob_ri=vx_prob_ri,
                vx_prob_sia=vx_prob_sia,
                seed=rep,
                dur_exp=lp.constant(value=0),
                individual_heterogeneity=heterogeneity,
                init_immun_scalar=init_immun_scalar,
                r0_scalar_wt_slope=r0_scalar_wt_slope,
                r0_scalar_wt_intercept=r0_scalar_wt_intercept,
                ipv_vx=ipv_vx,
                verbose=0,
            )

            # Save the results
            I_series_store[(heterogeneity, r0)] = sim.results.I_by_strain[:, 0, 0]  # all times, single node, VDPV2 strain
            new_exposed = sim.results.new_exposed_by_strain[:, 0, 0]  # all times, single node, VDPV2 strain
            new_exposed_store[(heterogeneity, r0)] = new_exposed
            records.append(
                {
                    "r0": r0,
                    "heterogeneity": heterogeneity,
                    "init_prev": init_prev,
                    "rep": rep,
                    "total_new_exposed": new_exposed.sum(),
                }
            )


# Convert records to DataFrame
df = pd.DataFrame.from_records(records)
df["prop_infected"] = df["total_new_exposed"] / init_pop
df.to_csv(results_path + "/prop_infected.csv", index=False)
grouped = df.groupby(["r0", "heterogeneity"])["prop_infected"].mean().reset_index()
merged = grouped.merge(output[["r0", "I_inf_exp"]], left_on="r0", right_on="r0", how="inner")
merged["delta"] = merged["prop_infected"] - merged["I_inf_exp"]


# ----- Plotting -----#

styles = {
    (True): ("Hetero", "tab:green", "solid"),
    (False): ("NoHetero", "tab:blue", "solid"),
}


# Plot the Infection time series
timeseries_path = Path(results_path) / "timeseries_of_new_exposed"
os.makedirs(timeseries_path, exist_ok=True)
for r0 in r0_values:
    plt.figure(figsize=(10, 6))
    for (hetero), (label, color, style) in styles.items():
        key = (hetero, r0)
        if key in I_series_store:
            I_t = I_series_store[key]
            plt.plot(I_t, label=label, color=color, linestyle=style)
    plt.title(f"Infected Over Time at r0 = {r0}")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Infected Individuals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(timeseries_path / f"infection_time_series_r0_{r0}.png")
    # plt.show()
    plt.close()


# Plot the proportion infected vs r0
plt.figure(figsize=(10, 6))
for (hetero), (label, color, style) in styles.items():
    subset = grouped[(grouped["heterogeneity"] == hetero)]
    plt.plot(subset["r0"], subset["prop_infected"], label=label, color=color, linestyle=style)
plt.plot(output["r0"], output["I_inf_exp"], "k--", label="Expected (KM)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.legend()
plt.xlabel(r"$R_0$")
plt.ylabel("Proportion Infected")
plt.title("Final Epidemic Size vs $R_0$ Across Infection Methods")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(Path(results_path) / "prop_infected_vs_r0_all_configs.png")
# plt.show()
plt.close()


# Plot the difference between expected and simulated
plt.figure(figsize=(10, 6))
for (hetero), (label, color, style) in styles.items():
    subset = merged[(merged["heterogeneity"] == hetero)]
    plt.plot(subset["r0"], subset["delta"], label=label, color=color, linestyle=style)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel(r"$R_0$")
plt.ylabel("Observed - Expected (Proportion Infected)")
plt.title("Error vs Expected Final Size (KM)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(Path(results_path) / "diff_vs_r0_all_configs.png")
# plt.show()
plt.close()

sc.printcyan("Done.")
