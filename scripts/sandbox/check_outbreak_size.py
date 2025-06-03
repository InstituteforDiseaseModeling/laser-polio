import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
from scipy.optimize import fsolve

import laser_polio as lp

# Based on: https://github.com/InstituteforDiseaseModeling/laser-generic/blob/main/notebooks/04_SIR_nobirths_outbreak_size.ipynb

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA:ANKA"]
start_year = 2019
n_days = 365 * 2
pop_scale = 1 / 1
init_region = "ANKA"
results_path = "results/check_outbreak_size"
n_reps = 1
# r0_values = np.linspace(1, 2, 2)
r0_values = np.linspace(0, 10, 15)
n_ppl = 1e6
init_prev = 20 / n_ppl
S0 = 1.0

######### END OF USER PARS ########
###################################

os.makedirs(results_path, exist_ok=True)


# Calculate the expected final size using the Kermack-McKendrick model
def KM_limit(z, R0, S0, I0):
    if R0 * S0 < 1:
        return 0
    else:
        return z - S0 * (1 - np.exp(-R0 * (z + I0)))


# Expected
population = n_ppl
inf_mean = 24
init_inf = 20
# R0s = np.concatenate((np.linspace(0.2, 1.0, 5), np.linspace(1.5, 10.0, 25)))
S0s = [1.0]
output = pd.DataFrame(list(itertools.product(r0_values, S0s)), columns=["R0", "S0"])
output["I_inf_exp"] = [
    fsolve(KM_limit, 0.5 * (R0 * S0 >= 1), args=(R0, S0, init_inf / population))[0]
    for R0, S0 in zip(output["R0"], output["S0"], strict=False)
]
output["S_inf_exp"] = output["S0"] - output["I_inf_exp"]


# Simulated
I_series_store = {}  # Key: (heterogeneity, infect_method, r0), Value: 1D array of I over time
records = []
for r0 in r0_values:
    print(f"\nSweeping R0 = {r0:.2f}")

    for heterogeneity, infect_method in itertools.product([True, False], ["classic", "fast"]):
        label = f"{'Hetero' if heterogeneity else 'NoHetero'}-{infect_method}"
        print(f" → Config: {label}")

        for rep in range(n_reps):
            sim = lp.run_sim(
                regions=regions,
                start_year=start_year,
                n_days=n_days,
                pop_scale=pop_scale,
                init_region=init_region,
                init_prev=init_prev,
                results_path=results_path,
                save_plots=False,
                save_data=False,
                n_ppl=n_ppl,
                r0=r0,
                init_immun=[0.0],
                seasonal_amplitude=0.0,
                cbr=np.array([0]),
                vx_prob_ri=None,
                vx_prob_sia=None,
                seed=rep,
                dur_exp=lp.constant(value=2),
                individual_heterogeneity=heterogeneity,
                infection_method=infect_method,
            )

            last_non_zero_R = np.where(sim.results.R[:, 0] > 0)[0][-1]
            final_R = np.sum(sim.results.R[last_non_zero_R])

            # Save the results
            I_series_store[(heterogeneity, infect_method, r0)] = np.sum(sim.results.I, axis=1)
            records.append(
                {
                    "r0": r0,
                    "heterogeneity": heterogeneity,
                    "infect_method": infect_method,
                    "init_prev": init_prev,
                    "rep": rep,
                    "final_recovered": final_R,
                }
            )


# Convert records to DataFrame
df_results = pd.DataFrame.from_records(records)
df_results["prop_infected"] = df_results["final_recovered"] / n_ppl
df_results.to_csv(results_path + "/prop_infected.csv", index=False)
grouped = df_results.groupby(["r0", "heterogeneity", "infect_method"])["prop_infected"].mean().reset_index()
merged = grouped.merge(output[["R0", "I_inf_exp"]], left_on="r0", right_on="R0", how="inner")
merged["delta"] = merged["prop_infected"] - merged["I_inf_exp"]


# ----- Plotting -----#

styles = {
    (True, "classic"): ("Hetero + Classic", "tab:green", "solid"),
    (True, "fast"): ("Hetero + Fast", "tab:green", "dashed"),
    (False, "classic"): ("NoHetero + Classic", "tab:blue", "solid"),
    (False, "fast"): ("NoHetero + Fast", "tab:blue", "dashed"),
}


# Plot the Infection time series
target_r0 = 3.0
for r0 in r0_values:
    plt.figure(figsize=(10, 6))
    for (hetero, method), (label, color, style) in styles.items():
        key = (hetero, method, r0)
        if key in I_series_store:
            I_t = I_series_store[key]
            plt.plot(I_t, label=label, color=color, linestyle=style)
    plt.title(f"Infected Over Time at R0 = {r0}")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Infectious Individuals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(results_path) / f"infection_time_series_r0_{r0}.png")
    # plt.show()


# Plot the proportion infected vs R0
plt.figure(figsize=(10, 6))
for (hetero, method), (label, color, style) in styles.items():
    subset = grouped[(grouped["heterogeneity"] == hetero) & (grouped["infect_method"] == method)]
    plt.plot(subset["r0"], subset["prop_infected"], label=label, color=color, linestyle=style)
plt.plot(output["R0"], output["I_inf_exp"], "k--", label="Expected (KM)")
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


# Plot the difference between expected and simulated
plt.figure(figsize=(10, 6))
for (hetero, method), (label, color, style) in styles.items():
    subset = merged[(merged["heterogeneity"] == hetero) & (merged["infect_method"] == method)]
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


sc.printcyan("Done.")
