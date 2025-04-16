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
results_path = "results/scan_over_seasonality_zamfara"
# Define the range of par values to sweep
n_pts = 5  # Number of points to simulate
n_reps = 3
seasonal_factor_values = np.linspace(0, 1, n_pts)
seasonal_phase_values = np.linspace(1.0, 364.9, n_pts)


######### END OF USER PARS ########
###################################


# Create result matrices
total_infected_matrix = np.zeros((len(seasonal_phase_values), len(seasonal_factor_values)))
num_nodes_infected_matrix = np.zeros_like(total_infected_matrix)


# Run sweep
for i, seasonal_phase in enumerate(seasonal_phase_values):
    for j, seasonal_factor in enumerate(seasonal_factor_values):
        total_infected_accum = 0.0
        nodes_infected_accum = 0.0

        print(f"\nRunning {n_reps} reps for R0 = {seasonal_factor:.2f}, seasonal_phase = {seasonal_phase:.2f}")

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
                seasonal_factor=seasonal_factor,
                seasonal_phase=seasonal_phase,
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
    title="Total Infected vs seasonal_factor and seasonal_phase",
    filename="total_infected_heatmap_avg.png",
    xlabel="seasonal_factor",
    ylabel="seasonal_phase",
    xticks=seasonal_factor_values,
    yticks=seasonal_phase_values,
)

plot_heatmap(
    num_nodes_infected_matrix,
    title="Number of Nodes Infected vs seasonal_factor and seasonal_phase",
    filename="nodes_infected_heatmap_avg.png",
    xlabel="seasonal_factor",
    ylabel="seasonal_phase",
    xticks=seasonal_factor_values,
    yticks=seasonal_phase_values,
)

sc.printcyan("Sweep complete.")
