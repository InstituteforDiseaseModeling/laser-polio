import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer

import laser_polio as lp

# --- Setup the base network ---

regions = ["ZAMFARA"]
start_year = 2019
n_days = 180
results_path = "results/demo_sparsify"

# Get the regional data
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
n_nodes = len(dot_names)
node_lookup = lp.get_node_lookup("data/node_lookup.json", dot_names)
df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
df_comp = df_comp[df_comp["year"] == start_year]
pop = df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values
node_ids = sorted(node_lookup.keys())
lats = np.array([node_lookup[i]["lat"] for i in node_ids])
lons = np.array([node_lookup[i]["lon"] for i in node_ids])

# Generate distance matrix
n_nodes = len(node_ids)
dist_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(n_nodes):
        dist_matrix[i, j] = distance(lats[i], lons[i], lats[j], lons[j])


# --- Setup functions ---


def sparsify_network(network, keep_frac=0.2):
    """Randomly zero out some destination columns in the migration network."""

    net = network.copy()
    num_nodes = net.shape[1]
    keep = np.random.rand(num_nodes) < keep_frac
    net[:, ~keep] = 0  # zero out all flows into these destinations
    return net


# ---
gravity_k = 1e-12
gravity_c = 1.0
max_migr_frac = 1.0
net = gravity(pops=pop, distances=dist_matrix, k=gravity_k, a=1.0, b=1.0, c=gravity_c)
net /= np.power(pop.sum(), gravity_c)  # Normalize
net = row_normalizer(net, max_migr_frac)
net_sparsified = sparsify_network(net, keep_frac=0.2)

# Sum columns to get the total migration rate into each node
net_sum = net.sum(axis=0)
net_sparsified_sum = net_sparsified.sum(axis=0)

# Plot the original and sparsified networks
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
# Plot original network
axs[0].bar(range(n_nodes), net_sum, color="steelblue", alpha=0.7)
axs[0].set_title("Original Network")
axs[0].set_xlabel("Node Index")
axs[0].set_ylabel("Total Migration Rate")
axs[0].grid(True, linestyle="--", alpha=0.3)
# Plot sparsified network
axs[1].bar(range(n_nodes), net_sparsified_sum, color="darkorange", alpha=0.7)
axs[1].set_title("Sparsified Network")
axs[1].set_xlabel("Node Index")
axs[1].grid(True, linestyle="--", alpha=0.3)
# Overall figure title and layout
fig.suptitle("Total Migration Rate into Each Node", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(results_path, "sparsified_network.png"))
plt.show()


# --- Sweep over the migration parameters ---
gravity_k_values = np.array([1e-13, 1e-12])  # np.linspace(1, 10000, n_pts)
gravity_c_values = np.linspace(0.8, 1.2)
max_migr_frac = 1.0  # Maximum migration fraction
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

print("Done")
