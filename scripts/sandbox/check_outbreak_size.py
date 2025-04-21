import copy
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
results_path = "results/check_outbreak_size_fast_infect_with_heterogeneity"
n_reps = 1
# r0_values = np.linspace(1, 2, 2)
r0_values = np.linspace(0, 10, 15)
n_ppl = 1e6
init_prev = 20 / n_ppl
S0 = 1.0

######### END OF USER PARS ########
###################################

os.makedirs(results_path, exist_ok=True)


def KM_limit(z, R0, S0, I0):
    if R0 * S0 < 1:
        return 0
    else:
        return z - S0 * (1 - np.exp(-R0 * (z + I0)))


def plot_infected_by_node_dual(sim_hetero, sim_no_hetero, save=True, results_path=None):
    """
    Plot infected population over time for both simulations: with and without heterogeneity.
    """
    plt.figure(figsize=(10, 6))
    r0 = sim_hetero.pars.r0
    # Plot infection curves (total I over all nodes)
    I_hetero = np.sum(sim_hetero.results.I, axis=1)
    I_no_hetero = np.sum(sim_no_hetero.results.I, axis=1)
    plt.plot(I_hetero, label="With Heterogeneity", color="tab:green")
    plt.plot(I_no_hetero, label="No Heterogeneity", color="tab:blue", linestyle="--")
    plt.title(f"Total Infected Over Time (R0 = {r0:.2f})")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Number of Infectious Individuals")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    results_path = Path(results_path)
    if save:
        plt.savefig(results_path / f"infected_curve_R0_{r0:.2f}.png")
    else:
        plt.show()


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
    sim_hetero = None
    sim_no_hetero = None
    for hetergeneity in [True, False]:
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
                individual_heterogeneity=hetergeneity,
            )

            # Since we're ending the sim early & there's no init_immunity, the results values are 0 after the sim ends
            # So we need to find the last non-zero value in the results
            last_non_zero_R = np.where(sim.results.R[:, 0] > 0)[0][-1]
            final_R = np.sum(sim.results.R[last_non_zero_R])

            # Record the result
            records.append(
                {
                    "r0": r0,
                    "heterogeneity": hetergeneity,
                    "init_prev": init_prev,
                    "rep": rep,
                    "final_recovered": final_R,
                }
            )
        if hetergeneity:
            sim_hetero = copy.deepcopy(sim)
        else:
            sim_no_hetero = copy.deepcopy(sim)

    plot_infected_by_node_dual(sim_hetero, sim_no_hetero, save=True, results_path=results_path)


# Convert to DataFrame
df_results = pd.DataFrame.from_records(records)
df_results["prop_infected"] = df_results["final_recovered"] / n_ppl
df_results.to_csv(results_path + "/prop_infected.csv", index=False)


# ----- Plotting -----#

# Plot the proportion infected vs R0
grouped = df_results.groupby(["r0", "heterogeneity"])["prop_infected"].mean().reset_index()
plt.figure(figsize=(10, 6))
# Split by heterogeneity
for hetero_val, label, style in [(False, "No Heterogeneity", "tab:blue"), (True, "With Heterogeneity", "tab:green")]:
    subset = grouped[grouped["heterogeneity"] == hetero_val]
    plt.plot(subset["r0"], subset["prop_infected"], label=label, color=style)
    plt.plot(
        df_results[df_results["heterogeneity"] == hetero_val]["r0"],
        df_results[df_results["heterogeneity"] == hetero_val]["prop_infected"],
        ".",
        color=style,
        alpha=0.4,
        label="_nolegend_",
    )
# Plot expected values
plt.plot(output["R0"], output["I_inf_exp"], "k--", label="Expected (KM)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.legend()
plt.xlabel(r"$R_0$")
plt.ylabel(f"Proportion infected (after {n_days} days)")
plt.title("Outbreak Size vs $R_0$ with and without Heterogeneity")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(Path(results_path) / "prop_infected_vs_r0_hetero_comparison.png")
plt.show()


# plt.figure(figsize=(8, 6))
# colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
# color = next(colors)
# # # Get rows for this init_prev
# plt.plot(df_results["r0"], df_results["prop_infected"], ".", color=color, label="_nolegend_")
# plt.plot(grouped["r0"], grouped["prop_infected"], "-", color=color, label=f"S(0) = {1.0}")
# plt.plot(
#     output["R0"],
#     output["I_inf_exp"],
#     "k--",
#     label="Expected (KM)",
# )
# plt.legend()
# plt.xlabel(r"$R_{0}$")
# plt.ylabel(f"Proportion pop infected (after {n_days} days)")
# plt.grid(True)
# plt.tight_layout()
# plt.ylim(0, 1)
# plt.savefig(results_path + "/prop_infected_vs_r0_acq_trans_1.png")
# plt.show()


# Plot the difference between expected and simulated
merged = grouped.merge(output[["R0", "I_inf_exp"]], left_on="r0", right_on="R0", how="inner")
merged["delta"] = merged["prop_infected"] - merged["I_inf_exp"]  # Compute difference
for hetero_val, label, color in [(False, "No Heterogeneity", "tab:blue"), (True, "With Heterogeneity", "tab:green")]:
    subset = merged[merged["heterogeneity"] == hetero_val]
    plt.plot(subset["r0"], subset["delta"], "o-", label=label, color=color)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel(r"$R_0$")
plt.ylabel("Observed - Expected (proportion infected)")
plt.title("Difference between observed and expected final size vs $R_0$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(Path(results_path) / "diff_vs_r0_by_heterogeneity.png")
plt.show()

sc.printcyan("Done.")
