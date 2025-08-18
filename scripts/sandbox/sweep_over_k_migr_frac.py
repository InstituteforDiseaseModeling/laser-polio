import os

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2018
n_days = 180
pop_scale = 1 / 10
init_region = "ANKA"
init_prev = 0.01
results_path = "results/scan_over_k_migr_frac_zamfara"
# Define the range of par values to sweep
n_pts = 5  # Number of points to simulate
n_reps = 3
r0_values = [5, 10, 15]  # R0 values to sweep over
radiation_k_log10_values = np.linspace(-3.5, 1.0, n_pts)
max_migr_frac_values = np.linspace(0.0, 0.1, n_pts)


######### END OF USER PARS ########
###################################


# Plot heatmaps
os.makedirs(results_path, exist_ok=True)

# Run sweep for each R0 value
for r0 in r0_values:
    print(f"\n{'=' * 50}")
    print(f"SWEEPING OVER R0 = {r0}")
    print(f"{'=' * 50}")

    # Create result matrices for this R0
    total_infected_matrix = np.zeros((len(radiation_k_log10_values), len(max_migr_frac_values)))
    num_nodes_infected_matrix = np.zeros_like(total_infected_matrix)

    # Run sweep
    for i, radiation_k_log10 in enumerate(radiation_k_log10_values):
        for j, max_migr_frac in enumerate(max_migr_frac_values):
            total_infected_accum = 0.0
            nodes_infected_accum = 0.0

            print(
                f"\nRunning {n_reps} reps for R0 = {r0}, radiation_k_log10 = {radiation_k_log10:.2f}, max_migr_frac = {max_migr_frac:.3f}"
            )

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
                    radiation_k_log10=radiation_k_log10,
                    max_migr_frac=max_migr_frac,
                    seed=rep,  # Optional: control randomness
                )

                total_infected = sim.results.I_by_strain[:, :, 0].sum()
                num_nodes_infected = np.sum(sim.results.I_by_strain[:, :, 0].sum(axis=0) > 0)

                total_infected_accum += total_infected
                nodes_infected_accum += num_nodes_infected

            # Store averages
            total_infected_matrix[i, j] = total_infected_accum / n_reps
            num_nodes_infected_matrix[i, j] = nodes_infected_accum / n_reps

    # Create combined figure with both heatmaps
    def plot_combined_heatmaps(total_infected_matrix, num_nodes_infected_matrix, r0_value, xticks, yticks):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot total infected
        im1 = ax1.imshow(total_infected_matrix, origin="lower", cmap="viridis", aspect="auto")
        ax1.set_xticks(ticks=np.arange(len(xticks)))
        ax1.set_xticklabels([f"{x:.3f}" for x in xticks])
        ax1.set_yticks(ticks=np.arange(len(yticks)))
        ax1.set_yticklabels([f"{y:.1f}" for y in yticks])
        ax1.set_xlabel("max_migr_frac")
        ax1.set_ylabel("radiation_k_log10")
        ax1.set_title("Total Infected")
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("Total Infected")

        # Plot nodes infected
        im2 = ax2.imshow(num_nodes_infected_matrix, origin="lower", cmap="plasma", aspect="auto")
        ax2.set_xticks(ticks=np.arange(len(xticks)))
        ax2.set_xticklabels([f"{x:.3f}" for x in xticks])
        ax2.set_yticks(ticks=np.arange(len(yticks)))
        ax2.set_yticklabels([f"{y:.1f}" for y in yticks])
        ax2.set_xlabel("max_migr_frac")
        ax2.set_ylabel("radiation_k_log10")
        ax2.set_title("Number of Nodes Infected")
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label("Nodes Infected")

        # Add super title
        fig.suptitle(f"Disease Spread vs radiation_k_log10 and max_migr_frac (R₀ = {r0_value})", fontsize=16, y=0.98)
        plt.tight_layout()

        # Save with R0 in filename
        filename = f"heatmap_of_total_infected_and_nodes_infected_with_R0_{r0_value}.png"
        plt.savefig(os.path.join(results_path, filename), dpi=300, bbox_inches="tight")
        plt.show()

    plot_combined_heatmaps(total_infected_matrix, num_nodes_infected_matrix, r0, max_migr_frac_values, radiation_k_log10_values)

sc.printcyan("Sweep complete.")
