import os

import numpy as np
import pandas as pd
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA:ANKA"]
start_year = 2019
n_days = 180
pop_scale = 1 / 1
init_region = "ANKA"
# init_prev = 0.01
results_path = "results/check_outbreak_size"
# Define the range of par values to sweep
n_reps = 1
r0_values = np.linspace(0, 10, 6)
init_prev_values = np.linspace(0.2, 1.0, 5)

# TODO
# - Will have to count up recovereds not I
# - Make age pyramid for soley u15
# - think about dur_exp

######### END OF USER PARS ########
###################################


# Create result matrices
# total_infected_matrix = np.zeros((len(init_prev_values), len(r0_values)))
records = []


# Run sweep
for i, init_prev in enumerate(init_prev_values):
    for j, r0 in enumerate(r0_values):
        total_infected_accum = 0.0

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
                n_ppl=1000,
                r0=r0,
                init_immun=[0.0],
                seasonal_factor=0.0,
                cbr=np.array([0]),
                vx_prob_ri=None,
                vx_prob_sia=None,
                seed=rep,  # Optional: control randomness
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

# Save or analyze
os.makedirs(results_path, exist_ok=True)
df_results.to_csv(results_path + "/sweep_results.csv", index=False)

# Plot heatmaps
import itertools

import matplotlib.pyplot as plt

# Grouped mean for lines
grouped = df_results.groupby(["init_prev", "r0"])["final_recovered"].mean().reset_index()

plt.figure(figsize=(8, 6))
colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])

# Get unique init_prev values to loop over
init_prev_values = sorted(df_results["init_prev"].unique())

for init_prev in init_prev_values:
    color = next(colors)

    # Get rows for this init_prev
    scatter_vals = df_results[df_results["init_prev"] == init_prev]
    line_vals = grouped[grouped["init_prev"] == init_prev]

    # Scatter individual points
    plt.plot(scatter_vals["r0"], scatter_vals["final_recovered"], ".", color=color, label="_nolegend_")

    # Plot the average line
    plt.plot(line_vals["r0"], line_vals["final_recovered"], "-", color=color, label=f"S(0) = {1.0 - init_prev:.1f}")

plt.xlabel(r"$R_{0}$")
plt.ylabel(r"Final recovered $R(\infty)$")
plt.title("Final Recovered vs $R_0$")
plt.legend(title="Initial Susceptible Fraction")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure()
colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
for i, init_prev in enumerate(init_prev_values):
    vals = total_infected_matrix[i, :]
    color = next(colors)
    plt.plot(output[condition]["R0"], output[condition]["I_inf_exp"], color=color)
    plt.plot(output[condition]["R0"], output[condition]["I_inf_obs"], ".", label="_nolegend_", color=color)
plt.xlabel("$R_{0}$")
plt.ylabel("$I(t \\rightarrow {\\infty})$")
plt.legend(["$S(0) = 1.0$", "$S(0)= 0.8$", "$S(0) = 0.6$", "$S(0) = 0.4$", "$S(0) = 0.2$"])
plt.figure()

# def plot_heatmap(matrix, title, filename, xlabel, ylabel, xticks, yticks):
#     plt.figure(figsize=(8, 6))
#     im = plt.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
#     plt.colorbar(im, label="Value")
#     plt.xticks(ticks=np.arange(len(xticks)), labels=[f"{x:.1f}" for x in xticks])
#     plt.yticks(ticks=np.arange(len(yticks)), labels=[f"{y:.1f}" for y in yticks])
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(os.path.join(results_path, filename))
#     plt.show()


# plot_heatmap(
#     total_infected_matrix,
#     title="Total Infected vs R₀ and gravity_k",
#     filename="total_infected_heatmap_avg.png",
#     xlabel="R₀",
#     ylabel="gravity_k",
#     xticks=r0_values,
#     yticks=gravity_k_values,
# )

# plot_heatmap(
#     num_nodes_infected_matrix,
#     title="Number of Nodes Infected vs R₀ and gravity_k",
#     filename="nodes_infected_heatmap_avg.png",
#     xlabel="R₀",
#     ylabel="gravity_k",
#     xticks=r0_values,
#     yticks=gravity_k_values,
# )

sc.printcyan("Sweep complete.")
