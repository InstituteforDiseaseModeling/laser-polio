import os

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2019
n_days = 180
pop_scale = 1 / 10
init_region = "ANKA"
init_prev = 0.01
results_path = "results/scan_over_r0_k_zamfara"
# Define the range of par values to sweep
n_pts = 5  # Number of points to simulate
n_reps = 3
r0_values = np.linspace(5, 50, n_pts)
gravity_k_values = np.linspace(1.0, 100.0, n_pts)


######### END OF USER PARS ########
###################################


# Create result matrices
total_infected_matrix = np.zeros((len(gravity_k_values), len(r0_values)))
num_nodes_infected_matrix = np.zeros_like(total_infected_matrix)


# Run sweep
for i, gravity_k in enumerate(gravity_k_values):
    for j, r0 in enumerate(r0_values):
        total_infected_accum = 0.0
        nodes_infected_accum = 0.0

        print(f"\nRunning {n_reps} reps for R0 = {r0:.2f}, gravity_k = {gravity_k:.2f}")

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
                r0=r0,
                gravity_k=gravity_k,
                seed=rep,  # Optional: control randomness
            )

            total_infected = sim.results.I.sum()
            num_nodes_infected = np.sum(sim.results.I.sum(axis=0) > 0)

            total_infected_accum += total_infected
            nodes_infected_accum += num_nodes_infected

        # Store averages
        total_infected_matrix[i, j] = total_infected_accum / n_reps
        num_nodes_infected_matrix[i, j] = nodes_infected_accum / n_reps


# Plot heatmaps
os.makedirs(results_path, exist_ok=True)


def plot_heatmap(matrix, title, filename, xlabel, ylabel, xticks, yticks):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Value")
    plt.xticks(ticks=np.arange(len(xticks)), labels=[f"{x:.1f}" for x in xticks])
    plt.yticks(ticks=np.arange(len(yticks)), labels=[f"{y:.1f}" for y in yticks])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, filename))
    plt.show()


plot_heatmap(
    total_infected_matrix,
    title="Total Infected vs R₀ and gravity_k",
    filename="total_infected_heatmap_avg.png",
    xlabel="R₀",
    ylabel="gravity_k",
    xticks=r0_values,
    yticks=gravity_k_values,
)

plot_heatmap(
    num_nodes_infected_matrix,
    title="Number of Nodes Infected vs R₀ and gravity_k",
    filename="nodes_infected_heatmap_avg.png",
    xlabel="R₀",
    ylabel="gravity_k",
    xticks=r0_values,
    yticks=gravity_k_values,
)

sc.printcyan("Sweep complete.")
