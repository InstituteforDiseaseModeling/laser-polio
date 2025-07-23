import numpy as np
import pandas as pd
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from matplotlib import pyplot as plt

import laser_polio as lp

# USER PARS
regions = ["NIGERIA"]
start_year = 2018
pop_scale = 1 / 1  # Scale factor for population
max_migr_frac = 1.0  # Maximum migration fraction
k = 10**-8.896082688911651
a = 1.048239808810217
b = 0.7554399511356332
c = 0.459539396713007

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
net = gravity(pop, dist_matrix, k, a, b, c)
net /= np.power(pop.sum(), c)  # Normalize
net = net.astype(float)
net = row_normalizer(net, max_migr_frac)

# ----- Check the number of nodes that have hit the max migration fraction -----

rowsums = net.sum(axis=1)
n_maxed = np.sum(rowsums >= max_migr_frac)  # Number of rows that would be capped
prop_maxed = n_maxed / n_nodes  # Normalize by number of nodes


# Plot a histogram of the row sums
plt.figure(figsize=(8, 6))
plt.hist(rowsums, bins=100)
plt.title("Histogram of row sums")
plt.xlabel("Row sum")
plt.ylabel("Count")
plt.grid(True)
plt.savefig("results/gravity_nigeria_best_calib_row_sums.png")
plt.show()

print(f"{n_maxed} of {n_nodes} rows ({prop_maxed:.2%}) exceed max_migr_frac={max_migr_frac}")

plt.figure(figsize=(8, 6))
plt.hist(rowsums, bins=10)
# plt.hist(np.random.uniform(0, 1, 774), bins=10)
plt.axvline(max_migr_frac, color="red", linestyle="--", label="max_migr_frac")
plt.title("Histogram of Migration Matrix Row Sums")
plt.xlabel("Row sum")
plt.ylabel("Count")
plt.grid(True)
# Set the x-axis to be from 0 to 1
plt.xlim(0, 1)
plt.legend()
plt.savefig("results/gravity_nigeria_best_calib_row_sums.png")
plt.show()


print("Done.")
