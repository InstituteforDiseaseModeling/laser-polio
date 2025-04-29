import numpy as np
import pandas as pd
from laser_core.migration import distance
from laser_core.migration import radiation
from laser_core.migration import row_normalizer
from matplotlib import pyplot as plt

import laser_polio as lp

# USER PARS
regions = ["NIGERIA"]
start_year = 2018
pop_scale = 1 / 1  # Scale factor for population
max_migr_frac = 1.0  # Maximum migration fraction
k_values = np.linspace(0, 3, 50)  # Sweep k from 10 to 1000 over 100 points

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
# print("Distance matrix (km):")
# print(np.round(dist_matrix, 2))

# Sweep across k values
nets = np.zeros((len(k_values), n_nodes, n_nodes))  # Track networks
for idx, k in enumerate(k_values):
    net = radiation(pop, dist_matrix, k=k, include_home=False)
    net = net.astype(float)
    net = row_normalizer(net, max_migr_frac)
    nets[idx] = net  # Save the full network for this k

# # Plotting
# # Setup grid
# n_nodes = nets.shape[1]
# ncols = 4
# nrows = int(np.ceil(n_nodes / ncols))
# fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), constrained_layout=True)
# axs = axs.flatten()
# for i in range(n_nodes):  # Origin node
#     ax = axs[i]
#     for j in range(n_nodes):  # Destination node
#         if i == j:
#             continue  # skip self-migration if you want
#         ax.plot(k_values, nets[:, i, j], label=f"to {j}")
#     ax.set_title(f"Origin node {i}")
#     ax.set_xlabel("k value")
#     ax.set_ylabel("Normalized flow")
#     ax.grid(True)
#     # ax.legend(fontsize=6, loc="upper right")  # Small font for readability
# # Hide unused subplots
# for idx in range(n_nodes, len(axs)):
#     axs[idx].axis("off")
# plt.suptitle(f"Radiation sweep across k - ZAMFARA - max_migr_frac = {max_migr_frac}", fontsize=22)
# plt.savefig("results/radiation_sweep_k_zamfara.png")
# plt.show()


# ----- Check the number of nodes that have hit the max migration fraction -----

maxed_rows = []  # Will store number of maxed-out rows for each k
row_sums_over_k = []  # Will track row sums at each k

for idx, k in enumerate(k_values):
    net = radiation(pop, dist_matrix, k=k, include_home=False)
    net = net.astype(float)

    # Before normalization: check row sums
    rowsums = net.sum(axis=1)
    n_maxed = np.sum(rowsums > max_migr_frac)  # Number of rows that would be capped
    maxed_rows.append(n_maxed)

    net = row_normalizer(net, max_migr_frac)
    rowsums = net.sum(axis=1)
    row_sums_over_k.append(rowsums)
    nets[idx] = net
prop_maxed = np.array(maxed_rows) / n_nodes  # Normalize by number of nodes
row_sums_over_k = np.stack(row_sums_over_k)  # shape = (len(k_values), n_nodes)

plt.figure(figsize=(8, 6))
plt.plot(k_values, prop_maxed, marker="o")
plt.title("Proportion of nodes at max migration fraction")
plt.xlabel("k value")
plt.ylabel("Proportion of nodes at max migration fraction")
plt.grid(True)
plt.savefig("results/radiation_sweep_k_nigeria_prop_maxed.png")
plt.show()


# Violin plot
step = 5  # Pick every 5th k to avoid overcrowding
selected_idxs = np.arange(0, len(k_values), step)
selected_row_sums = row_sums_over_k[selected_idxs]  # (n_selected_k, n_nodes)
# Now TRANSPOSE: we want node distributions per k
data_for_violin = selected_row_sums.T  # Now shape (n_nodes, n_selected_k)
fig, ax = plt.subplots(figsize=(14, 6))
parts = ax.violinplot(dataset=data_for_violin, positions=k_values[selected_idxs], showmeans=False, showextrema=True, showmedians=True)
ax.set_title("Radiation model - Nigeria - distribution of row sums across k", fontsize=16)
ax.set_xlabel("k value", fontsize=14)
ax.set_ylabel("Row Sum After Normalization", fontsize=14)
ax.axhline(max_migr_frac, linestyle="--", color="red", label=f"max_migr_frac={max_migr_frac}")
ax.legend()
ax.grid(True)
plt.savefig("results/radiation_sweep_k_nigeria_distribution_row_sums.png")
plt.show()


print("Done.")
