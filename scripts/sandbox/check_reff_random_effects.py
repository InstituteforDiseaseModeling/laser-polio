import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import laser_polio as lp

start_year = 2019
regions = ["NIGERIA"]


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


# Find the dot_names matching the specified string(s)
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

# Load the NEW curated random effects
df_re = pd.read_csv("data/curation_scripts/random_effects/random_effects_curated.csv")
# Filter to adm0_name that == "NIGERIA"
df_re = df_re[df_re["adm0_name"].isin(regions)]
# Extract the Reff values
reff_re = df_re["reff_random_effect"].values  # R random effects from regression model

# Load the OLD Nigeria R scalars from EMOD
json_path = Path("data/curation_scripts/random_effects/r0_NGA_mult_2025Feb.json")
with json_path.open("r") as f:
    emod_scalars_dict = json.load(f)
emod_scalars = np.array(list(emod_scalars_dict.values()))


# Set bounds on R0
R0 = 14
R_m = 3.41 / R0  # Min R0. Divide by R0 since we ultimately want scalars on R0 (e.g., r0_scalars)
R_M = 16.7 / R0  # Max R0. Divide by R0 since we ultimately want scalars on R0 (e.g., r0_scalars)

# Compute scale and center values for the old and new datasets
# The emod_scalars are scalars on R0, the reff_re are random effects from the regression model
pim_scale = np.std(reff_re)  # sd from polio immunity mapper Reff random effects  = 0.589
pim_center = np.median(reff_re)  # median from PIM = 0.741
emod_scale = np.std(np.log((emod_scalars - 0.2) / (1 - emod_scalars + 0.2)))  # sd of logit transformed emod_scalars = 2.485
emod_center = np.median(np.log((emod_scalars - 0.2) / (1 - emod_scalars + 0.2)))  # median of logit transformed emod_scalars = -1.050

# # Option 1:
# # new_R0 = exp(new_scale * (random_effect - pim_center) / pim_scale + new_center)
# new_R0_scalars = np.exp(new_scale * (reff_re - pim_center) / pim_scale + new_center)
# # This has some flaws. First, the median < mean with something exponentiated so the mean after doing exp() could look strange. Second, as you suggest you technically have an unbounded R0.

# Option 2: you might consider limits to how low or high R0 can go, e.g. max(min(R0, 15), 5), or a logit instead of log that uses these (or other) bounds.  This is not a smooth function, but:
w = inv_logit(emod_scale * (reff_re - pim_center) / pim_scale)
R_c = np.exp(emod_center)  # Nigeria central R0 scalar = 1.91

reff_scalars = R_c + (R_M - R_c) * np.maximum(w - 0.5, 0) * 2 + (R_c - R_m) * np.minimum(w - 0.5, 0) * 2
# With the idea that at w = 0.5 we are at the Nigeria center, as w -> 1, we get to the bound R_M, and as w ->0, we go to R_m.
mean_r0 = np.mean(reff_scalars) * R0
min_r0 = np.min(reff_scalars) * R0
max_r0 = np.max(reff_scalars) * R0
print(f"Our R0: {mean_r0:.2f} ({min_r0:.2f}, {max_r0:.2f})")  # R0: 27.54 (18.01, 31.88)
mean_kurt = np.mean(emod_scalars * 14)
min_kurt = np.min(emod_scalars * 14)
max_kurt = np.max(emod_scalars * 14)
print(f"Kurt's R0: {mean_kurt:.2f} ({min_kurt:.2f}, {max_kurt:.2f})")  # R0: 27.54 (18.01, 31.88)


# ------- Plotting ------

# Load the shapefile
shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm1")
shp = shp[shp["adm0_name"].isin(regions)]

# Convert emod_scalars_dict to a DataFrame
emod_scalars_df = pd.DataFrame(list(emod_scalars_dict.items()), columns=["dot_name", "emod_scalar"])
# Merge the DataFrame with the GeoDataFrame on the dot_name column
shp = shp.merge(emod_scalars_df, on="dot_name", how="left")

# Curate the transformed random effects
df_re["new_re"] = reff_scalars
df_re["dot_name"] = "AFRO:" + df_re["adm0_name"] + ":" + df_re["adm1_name"]

shp = shp.merge(df_re[["dot_name", "new_re"]], on="dot_name", how="left")


# Plot the choropleth
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
shp.plot(column="emod_scalar", cmap="viridis", legend=True, ax=ax)

# Add a title and labels
ax.set_title("Choropleth of emod_scalar", fontsize=16)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Show the plot
plt.show()


# Plot the choropleth
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
shp.plot(column="new_re", cmap="viridis", legend=True, ax=ax)

# Add a title and labels
ax.set_title("Choropleth of new_re_scalar", fontsize=16)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Show the plot
plt.show()
