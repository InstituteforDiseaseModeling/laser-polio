import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import laser_polio as lp

regions = ["NIGERIA"]

# Load the NEW curated random effects
df_re = pd.read_csv("data/curation_scripts/random_effects/random_effects_curated.csv")
df_re = df_re[df_re["adm0_name"].isin(regions)]
# Extract the random effects
sia_re = (
    df_re["sia_random_effect"].values * -1
)  # R random effects from regression model. Take negative since negative values are 'high' SIA coverage


def calc_sia_prob_from_rand_eff(sia_re, center=0.5, scale=0.8):
    """Convert SIA random effects to probabilities."""
    vals_rescaled = scale * sia_re + np.log(center / (1 - center))  # Center & scale the random effects (source = Hil???)
    sia_probs = lp.inv_logit(vals_rescaled)  # Convert to probabilities
    return sia_probs


center = 0.7
scale = 2.4
sia_prob = calc_sia_prob_from_rand_eff(sia_re, center=center, scale=scale)
print(f"SIA prob with center={center} & scale={scale}: {sia_prob.mean():.2f} ({sia_prob.min():.2f}, {sia_prob.max():.2f})")


# ------- Plotting ------

# Load the shapefile
shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm1")
shp = shp[shp["adm0_name"].isin(regions)]

# Curate the transformed random effects
df_re["sia_prob"] = sia_prob
df_re["dot_name"] = "AFRO:" + df_re["adm0_name"] + ":" + df_re["adm1_name"]

shp = shp.merge(df_re[["dot_name", "sia_prob"]], on="dot_name", how="left")


# Plot the choropleth
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
shp.plot(column="sia_prob", cmap="viridis", legend=True, ax=ax)

# Add a title and labels
ax.set_title(f"SIA prob with center={center} & scale={scale}", fontsize=16)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Show the plot
plt.show()
