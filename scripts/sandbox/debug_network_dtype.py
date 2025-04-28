import numpy as np
from laser_core.migration import radiation
from laser_core.migration import row_normalizer
from matplotlib import pyplot as plt

# ----- Radiation sweep across k - dist_mat dtype = int -----

# Setup
pops = np.array([99510, 595855, 263884])
dist = np.array([[0, 4, 66], [4, 0, 827], [66, 827, 0]], dtype=int)
max_migr_frac = 0.3

k_values = np.linspace(0, 5, 100)  # Sweep k from 10 to 1000 over 100 points

# Track networks
nets = np.zeros((len(k_values), 3, 3))
for idx, k in enumerate(k_values):
    net = radiation(pops, dist, k=k, include_home=False)
    net = net.astype(float)
    net = row_normalizer(net, max_migr_frac)
    nets[idx] = net  # Save the full network for this k

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        ax.plot(k_values, nets[:, i, j], label=f"from {i} to {j}")
        ax.set_title(f"Flow from {i} to {j}")
        ax.set_xlabel("k value")
        ax.set_ylabel("Normalized Flow")
        ax.grid(True)
plt.suptitle("Radiation sweep across k - dist_mat dtype = int", fontsize=18)
plt.savefig("results/radiation_sweep_k_int.png")
plt.show()


# ----- Radiation sweep across k - dist_mat dtype = float -----

# Setup
pops = np.array([99510, 595855, 263884])
dist = np.array([[0, 4, 66], [4, 0, 827], [66, 827, 0]], dtype=float)
max_migr_frac = 0.3

k_values = np.linspace(0, 5, 100)  # Sweep k from 10 to 1000 over 100 points

# Track networks
nets = np.zeros((len(k_values), 3, 3))

for idx, k in enumerate(k_values):
    net = radiation(pops, dist, k=k, include_home=False)
    net = net.astype(float)
    net = row_normalizer(net, max_migr_frac)
    nets[idx] = net  # Save the full network for this k

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        ax.plot(k_values, nets[:, i, j], label=f"from {i} to {j}")
        ax.set_title(f"Flow from {i} to {j}")
        ax.set_xlabel("k value")
        ax.set_ylabel("Normalized Flow")
        ax.grid(True)
plt.suptitle("Radiation sweep across k - dist_mat dtype = float", fontsize=18)
plt.savefig("results/radiation_sweep_k_float.png")
plt.show()

print("Done.")
