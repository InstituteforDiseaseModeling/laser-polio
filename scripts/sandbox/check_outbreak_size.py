import itertools
import os

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
n_days = 730
pop_scale = 1 / 1
init_region = "ANKA"
results_path = "results/check_outbreak_size"
n_reps = 1
r0_values = np.linspace(0, 10, 15)
n_ppl = 1e6
init_prev = 20 / n_ppl
S0 = 1.0

######### END OF USER PARS ########
###################################


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
records = []
for r0 in r0_values:
    print(f"\nRunning {n_reps} reps for {r0=:.2f}, {init_prev=:.2f}")

    for rep in range(n_reps):
        print(f"  ↳ Rep {rep + 1}/{n_reps}")

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
            seasonal_factor=0.0,
            cbr=np.array([0]),
            vx_prob_ri=None,
            vx_prob_sia=None,
            seed=rep,  # Optional: control randomness
            dur_exp=lp.constant(value=2),
        )

        final_R = np.sum(sim.results.R[-1])
        final_S = np.sum(sim.results.S[-1])

        # Record the result
        records.append(
            {
                "r0": r0,
                "init_prev": init_prev,
                "rep": rep,
                "final_recovered": final_R,
                "final_susceptible": final_S,
            }
        )

# Convert to DataFrame
df_results = pd.DataFrame.from_records(records)
df_results["prop_infected"] = df_results["final_recovered"] / n_ppl

# Save or analyze
os.makedirs(results_path, exist_ok=True)
df_results.to_csv(results_path + "/sweep_results.csv", index=False)


# ----- Plotting -----#

# Grouped mean for lines
grouped = df_results.groupby(["r0"])[["final_recovered", "prop_infected"]].mean().reset_index()
# grouped["prop_infected"] = grouped["final_recovered"] / n_ppl
plt.figure(figsize=(8, 6))
colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
color = next(colors)
# # Get rows for this init_prev
plt.plot(df_results["r0"], df_results["prop_infected"], ".", color=color, label="_nolegend_")
plt.plot(grouped["r0"], grouped["prop_infected"], "-", color=color, label=f"S(0) = {1.0}")
plt.plot(
    output["R0"],
    output["I_inf_exp"],
    "k--",
    label="Expected (KM)",
)
plt.legend()
plt.xlabel(r"$R_{0}$")
plt.ylabel(f"Proportion pop infected (after {n_days} days)")
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 1)
plt.savefig(results_path + "/prop_infected_vs_r0_acq_trans_1.png")
plt.show()

sc.printcyan("Done.")
