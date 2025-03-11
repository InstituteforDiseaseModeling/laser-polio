import matplotlib.pyplot as plt
import numpy as np

# Generate random lat/lon points
np.random.seed(42)
num_points = 100
lats = np.random.uniform(-10, 10, num_points)  # Random latitudes
lons = np.random.uniform(-10, 10, num_points)  # Random longitudes
infection_counts = np.random.randint(1, 100, num_points)  # Random infection counts

# List of colormaps to test
colormaps = ["viridis", "plasma", "inferno", "magma", "coolwarm", "cividis"]

# Create subplots for different colormaps
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axs = axs.ravel()

for i, cmap in enumerate(colormaps):
    scatter = axs[i].scatter(lons, lats, c=infection_counts, cmap=cmap, edgecolors="none", alpha=0.75)
    axs[i].set_title(f"Colormap: {cmap}")
    axs[i].set_xlabel("Longitude")
    axs[i].set_ylabel("Latitude")
    fig.colorbar(scatter, ax=axs[i], label="Infection Count")

plt.tight_layout()
plt.show()
