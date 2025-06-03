import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2019
n_days = 180
pop_scale = 1 / 10
init_region = "ANKA"
init_prev = 200
r0 = 14
results_path = "results/sweep_over_migration_gravity"
# Define the range of par values to sweep
n_pts = 4  # Number of points to simulate
n_reps = 3
gravity_k_values = np.array([1, 100, 1000, 10000])  # np.linspace(1, 10000, n_pts)
gravity_c_values = np.linspace(0.1, 1.5, n_pts)
max_migr_frac = 1.0  # Maximum migration fraction

######### END OF USER PARS ########
###################################


# --- First let's manally build the gravity network to understand the parameters ---

# Get the regional data
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
n_nodes = len(dot_names)
node_lookup = lp.get_node_lookup("data/node_lookup.json", dot_names)
df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
df_comp = df_comp[df_comp["year"] == start_year]
pop = df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values * pop_scale
node_ids = sorted(node_lookup.keys())
lats = np.array([node_lookup[i]["lat"] for i in node_ids])
lons = np.array([node_lookup[i]["lon"] for i in node_ids])

# Generate distance matrix
n_nodes = len(node_ids)
dist_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(n_nodes):
        dist_matrix[i, j] = distance(lats[i], lons[i], lats[j], lons[j])


# # Precompute log distances once (excluding diagonals)
# valid_mask = ~np.eye(n_nodes, dtype=bool)
# log_dist = np.log10(dist_matrix[valid_mask] + 1e-6)
# # Set up grid
# n_k = len(gravity_k_values)
# n_c = len(gravity_c_values)
# nets = {}
# for k_idx, k in enumerate(gravity_k_values):
#     for c_idx, c in enumerate(gravity_c_values):
#         net = gravity(
#             pops=pop,
#             distances=dist_matrix,
#             k=k,
#             a=1.0,
#             b=1.0,
#             c=c,
#         )
#         net = net.astype(float)
#         net /= np.power(pop.sum(), c)  # Normalize

#         net = row_normalizer(net, max_migr_frac)
#         row_sums = net.sum(axis=1)  # Total outflow from each node
#         print(f"(k={k}, c={c}) Row sums:", row_sums)
#         nets[(k, c)] = net  # Save using (k, c) as key


# --- Plot scatter of log(migration) vs log(distance) for each (k, c) pair ---

gravity_k_values = np.array([0.0000000000001, 0.000000000001, 0.00000000001, 0.0000000001])  # np.linspace(1, 10000, n_pts)
gravity_c_values = np.linspace(1, 2, 4)
# Precompute log distances once (excluding diagonals)
valid_mask = ~np.eye(n_nodes, dtype=bool)
log_dist = np.log10(dist_matrix[valid_mask] + 1e-6)
# Set up grid
n_k = len(gravity_k_values)
n_c = len(gravity_c_values)
fig, axes = plt.subplots(nrows=n_k, ncols=n_c, figsize=(4 * n_c, 3 * n_k), sharex=True, sharey=True)
axes = np.array(axes).reshape(n_k, n_c)
for i, k in enumerate(gravity_k_values):
    for j, c in enumerate(gravity_c_values):
        ax = axes[i, j]
        net = gravity(pops=pop, distances=dist_matrix, k=k, a=1.0, b=1.0, c=c)
        net /= np.power(pop.sum(), c)  # Normalize
        net = row_normalizer(net, max_migr_frac)
        mig_rates = net[valid_mask]
        log_mig = np.log10(mig_rates + 1e-12)
        ax.scatter(log_dist, log_mig, alpha=0.4, s=10)
        ax.set_title(f"k={k:.1e}, c={c:.2f}", fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if i == n_k - 1:
            ax.set_xlabel("log₁₀(Distance)")
        if j == 0:
            ax.set_ylabel("log₁₀(Migration Rate)")
fig.suptitle("log(Migration) vs log(Distance) across Gravity Parameters", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(results_path, "scatter_log_migration_vs_distance_grid.png"))
plt.show()


# # --- Plot rowsums for each (k, c) pair ---

# n_nodes = len(node_ids)  # or use pop.shape[0]
# ncols = 4
# nrows = int(np.ceil(n_nodes / ncols))
# for c in gravity_c_values:
#     fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), constrained_layout=True)
#     axs = axs.flatten()
#     for i in range(n_nodes):  # Origin node
#         ax = axs[i]
#         for j in range(n_nodes):  # Destination node
#             if i == j:
#                 continue  # skip self-migration
#             # Collect flow to dest j from origin i across all k (for this c)
#             flows = np.array([nets[(k, c)][i, j] for k in gravity_k_values])
#             ax.plot(gravity_k_values, flows, label=f"to {j}", alpha=0.6)
#         ax.set_title(f"Origin node {i}")
#         ax.set_xlabel("k value")
#         ax.set_ylabel("Normalized flow")
#         ax.grid(True)
#     # Clean up unused subplots if any
#     for i in range(n_nodes, len(axs)):
#         fig.delaxes(axs[i])
#     fig.suptitle(f"Flow from Each Origin to Destinations (c = {c:.2f})", fontsize=16)
#     outdir = os.path.join(results_path, "flow_vs_k_per_origin")
#     os.makedirs(outdir, exist_ok=True)
#     fig.savefig(os.path.join(outdir, f"flow_by_k_c{c:.2f}.png"))
#     plt.close(fig)
