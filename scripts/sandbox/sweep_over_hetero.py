import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2019
n_days = 180
pop_scale = 1
init_region = "ANKA"
init_prev = 200
r0 = 19.57
radiation_k = 0.049
seasonal_amplitude = 0.195
seasonal_peak_doy = 268
use_pim_scalars = True
results_path = "results/scan_over_heterogeneity"
# Define the range of par values to sweep
n_pts = 8  # Number of points to simulate
n_reps = 3
risk_mult_var_values = np.linspace(3.0, 12.0, n_pts)


######### END OF USER PARS ########
###################################


# Create result matrices
total_infected_matrix = np.zeros(len(risk_mult_var_values))
num_nodes_infected_matrix = np.zeros_like(total_infected_matrix)
acq_risk_values_by_riskmult = defaultdict(list)
infected_by_riskmult = {}  # key = risk_mult_var, value = sim.results.I

# Run sweep
for i, risk_mult_var in enumerate(risk_mult_var_values):
    total_infected_accum = 0.0
    nodes_infected_accum = 0.0

    print(f"\nRunning {n_reps} reps for risk_mult_var = {risk_mult_var:.2f}")

    for rep in range(n_reps):
        print(f"  ↳ Rep {rep + 1}/{n_reps}")

        sim = lp.run_sim(
            regions=regions,
            start_year=start_year,
            n_days=n_days,
            pop_scale=pop_scale,
            init_region=init_region,
            init_prev=init_prev,
            r0=r0,
            radiation_k=radiation_k,
            use_pim_scalars=use_pim_scalars,
            seasonal_amplitude=seasonal_amplitude,
            seasonal_peak_doy=seasonal_peak_doy,
            risk_mult_var=risk_mult_var,
            results_path=results_path,
            save_plots=False,
            save_data=False,
            run=False,
            seed=rep,  # Optional: control randomness
        )

        # Save the acquisition risk multipliers
        acq_risk_data = sim.people.acq_risk_multiplier[: sim.people.count]
        acq_risk_values_by_riskmult[risk_mult_var].extend(acq_risk_data.tolist())

        # Run the sim
        sim.run()

        # Store the results
        if risk_mult_var not in infected_by_riskmult:
            infected_by_riskmult[risk_mult_var] = sim.results.I.astype(np.float32).copy()
        else:
            infected_by_riskmult[risk_mult_var] += sim.results.I.astype(np.float32)
        total_infected = sim.results.I.sum()
        num_nodes_infected = np.sum(sim.results.I.sum(axis=0) > 0)

        total_infected_accum += total_infected
        nodes_infected_accum += num_nodes_infected

    # Store averages
    infected_by_riskmult[risk_mult_var] /= n_reps
    total_infected_matrix[i] = total_infected_accum / n_reps
    num_nodes_infected_matrix[i] = nodes_infected_accum / n_reps

os.makedirs(results_path, exist_ok=True)

# Plot histograms
fig, axes = plt.subplots(len(acq_risk_values_by_riskmult), 1, figsize=(8, 3 * len(acq_risk_values_by_riskmult)), sharex=True)
for ax, (risk_mult_var, values) in zip(axes, sorted(acq_risk_values_by_riskmult.items()), strict=False):
    ax.hist(values, bins=30, color="skyblue", edgecolor="black")
    ax.set_title(f"acq_risk_multiplier Histogram — risk_mult_var = {risk_mult_var:.2f}")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
axes[-1].set_xlabel("acq_risk_multiplier")
plt.tight_layout()
plt.savefig(f"{results_path}/acq_risk_histograms.png")
plt.show()

# Plot barplot for total_infected and num_nodes_infected
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].bar(risk_mult_var_values, total_infected_matrix, color="skyblue", edgecolor="black")
axes[0].set_title("Total Infected vs Risk Multiplier Variance")
axes[0].set_ylabel("Total Infected")
# axes[0].set_yscale("log")
axes[1].bar(risk_mult_var_values, num_nodes_infected_matrix, color="salmon", edgecolor="black")
axes[1].set_title("Number of Nodes Infected vs Risk Multiplier Variance")
axes[1].set_ylabel("Number of Nodes Infected")
axes[1].set_xlabel("Risk Multiplier Variance")
# axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig(f"{results_path}/plot_barplot_total_infected_and_nodes_infected.png")
plt.show()


def plot_infected_by_node_all_in_one(infected_by_riskmult, results_path=None, save=False):
    n_risks = len(infected_by_riskmult)
    ncols = 2
    nrows = (n_risks + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (risk_mult_var, I) in zip(axes, sorted(infected_by_riskmult.items()), strict=False):
        n_nodes = I.shape[1]
        for node in range(n_nodes):
            ax.plot(I[:, node], label=f"Node {node}")
        ax.set_title(f"acq_risk_multiplier = {risk_mult_var:.2f}")
        ax.set_xlabel("Time (Timesteps)")
        ax.set_ylabel("Infected")
        ax.grid(True)
        ax.legend(fontsize="xx-small", loc="upper right")

    # Hide any unused subplots
    for ax in axes[len(infected_by_riskmult) :]:
        ax.axis("off")

    plt.suptitle("Infected Over Time by Node — Grouped by Acquisition Risk Multiplier", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save and results_path:
        os.makedirs(results_path, exist_ok=True)
        plt.savefig(os.path.join(results_path, "infected_by_node_all_risks.png"))
        plt.close()
    else:
        plt.show()


plot_infected_by_node_all_in_one(infected_by_riskmult, results_path=results_path, save=True)


sc.printcyan("Sweep complete.")
