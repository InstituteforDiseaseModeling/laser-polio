import inspect

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import laser_polio as lp

start_year = 2019
# regions = ["AFRO", "EMRO"]
regions = ["NIGERIA"]


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def plot_choropleths_with_histograms(
    shp,
    var1_values,
    var2_values,
    var1_name=None,
    var2_name=None,
    cmap="viridis",
    colorbar_label="Value",
    save_path=None,
    dpi=300,
    show=True,
):
    """
    Plot side-by-side choropleths and histograms, optionally inferring variable names.

    Parameters:
    - shp: GeoDataFrame with geometry
    - var1_values, var2_values: arrays or Series of values to add and plot
    - var1_name, var2_name: optional names to use in the plots and GeoDataFrame
    """
    # Try to infer variable names if not provided
    if var1_name is None or var2_name is None:
        frame = inspect.currentframe().f_back
        arg_values = inspect.getargvalues(frame).locals
        for k, v in arg_values.items():
            if v is var1_values and var1_name is None:
                var1_name = k
            if v is var2_values and var2_name is None:
                var2_name = k

    if var1_name is None or var2_name is None:
        raise ValueError("Could not infer variable names. Please pass var1_name and var2_name explicitly.")

    # Add variables to the GeoDataFrame
    if len(var1_values) != len(shp) or len(var2_values) != len(shp):
        raise ValueError("Input variable lengths do not match shapefile length.")
    shp[var1_name] = var1_values
    shp[var2_name] = var2_values

    # Shared color scale
    vmin = min(shp[[var1_name, var2_name]].min())
    vmax = max(shp[[var1_name, var2_name]].max())

    # Precompute histogram y-limits
    hist1_counts, _ = np.histogram(shp[var1_name].dropna(), bins=20)
    hist2_counts, _ = np.histogram(shp[var2_name].dropna(), bins=20)
    hist_ylim = max(hist1_counts.max(), hist2_counts.max())

    # 2x2 layout: maps on top, histograms below
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=False, gridspec_kw={"height_ratios": [3, 1]})

    # Plot maps
    shp.plot(column=var1_name, ax=axes[0, 0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(var1_name)
    axes[0, 0].axis("off")

    shp.plot(column=var2_name, ax=axes[0, 1], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(var2_name)
    axes[0, 1].axis("off")

    # Plot histograms
    axes[1, 0].hist(shp[var1_name].dropna(), bins=20, color="gray", edgecolor="black")
    axes[1, 0].set_ylim(0, hist_ylim)
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(shp[var2_name].dropna(), bins=20, color="gray", edgecolor="black")
    axes[1, 1].set_ylim(0, hist_ylim)
    axes[1, 1].set_ylabel("Count")

    # Add colorbar
    fig.subplots_adjust(bottom=0.15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(colorbar_label)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ----- Load and curate the data -----

# Geography
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
shp = gpd.read_file(filename="data/shp_africa_low_res.gpkg", layer="adm2")
shp = shp[shp["dot_name"].isin(dot_names)]
shp.set_index("dot_name", inplace=True)
shp = shp.loc[dot_names].reset_index()

# Load the curated data
df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
df_comp = df_comp[df_comp["year"] == start_year]

# Extract the data
wt = df_comp.set_index("dot_name").loc[dot_names, "prop_underwt"].values
pim = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values

# Convert to R0 modifiers
r0_scalar_wt = (1 / (1 + np.exp(24 * (0.22 - wt)))) + 0.2  # The 0.22 is the mean of Nigeria underwt
mean_r0_scalar_wt = np.mean(r0_scalar_wt)
min_r0_scalar_wt = np.min(r0_scalar_wt)
max_r0_scalar_wt = np.max(r0_scalar_wt)

# ------ Transform the PIM random effects ------

# Option 1: Linear Rescaling (Min-Max Match)
pim_scaled = (pim - pim.min()) / (pim.max() - pim.min())  # Rescale to [0, 1]
r0_scalar_pim_ls = pim_scaled * (r0_scalar_wt.max() - r0_scalar_wt.min()) + r0_scalar_wt.min()
# Compute the mean, min, and max of the rescaled values
print(f"r0_scalar_wt: {mean_r0_scalar_wt:.2f} ({min_r0_scalar_wt:.2f}, {max_r0_scalar_wt:.2f})")
mean_r0_scalar_pim_ls = np.mean(r0_scalar_pim_ls)
min_r0_scalar_pim_ls = np.min(r0_scalar_pim_ls)
max_r0_scalar_pim_ls = np.max(r0_scalar_pim_ls)
print(f"pim_lr: {mean_r0_scalar_wt:.2f} ({min_r0_scalar_wt:.2f}, {max_r0_scalar_wt:.2f})")
# Plot maps of the original and rescaled values
plot_choropleths_with_histograms(shp, r0_scalar_wt, r0_scalar_pim_ls, save_path="results/plot_r0_scalars_underwt_pim_ls.png", show=True)

# Option 2: Match Mean and Range (Affine Transformation)
pim_centered = pim - pim.mean()
r = max_r0_scalar_wt - min_r0_scalar_wt  # Range of R0 wt
pim_scaled = pim_centered * (r / (pim.max() - pim.min()))
pim_mr = pim_scaled + mean_r0_scalar_wt
plot_choropleths_with_histograms(shp, r0_scalar_wt, pim_mr)
# # new_R0 = exp(new_scale * (random_effect - pim_center) / pim_scale + new_center)
# r0_re = np.exp(scale_wt * (re - center_re) / scale_re + center_wt)
# # # This has some flaws. First, the median < mean with something exponentiated so the mean after doing exp() could look strange. Second, as you suggest you technically have an unbounded R0.


# ----- Version just for Nigeria -----

regions = ["NIGERIA"]

# Geography
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
shp = gpd.read_file(filename="data/shp_africa_low_res.gpkg", layer="adm2")
shp = shp[shp["dot_name"].isin(dot_names)]
shp.set_index("dot_name", inplace=True)
shp = shp.loc[dot_names].reset_index()
# Load the curated data
df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
df_comp = df_comp[df_comp["year"] == start_year]
# Extract the data
wt = df_comp.set_index("dot_name").loc[dot_names, "prop_underwt"].values
pim = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values
# Convert to R0 modifiers
r0_scalar_wt = (1 / (1 + np.exp(24 * (0.22 - wt)))) + 0.2  # The 0.22 is the mean of Nigeria underwt
mean_r0_scalar_wt = np.mean(r0_scalar_wt)
min_r0_scalar_wt = np.min(r0_scalar_wt)
max_r0_scalar_wt = np.max(r0_scalar_wt)
# Option 1: Linear Rescaling (Min-Max Match)
nigeria_pim = pim
nig_min = nigeria_pim.min()
nig_max = nigeria_pim.max()
pim_scaled = (pim - nig_min) / (nig_max - nig_min)
# pim_scaled = (pim - pim.min()) / (pim.max() - pim.min())  # Rescale to [0, 1]
r0_scalar_pim_ls = pim_scaled * (r0_scalar_wt.max() - r0_scalar_wt.min()) + r0_scalar_wt.min()
# Compute the mean, min, and max of the rescaled values
print(f"r0_scalar_wt: {mean_r0_scalar_wt:.2f} ({min_r0_scalar_wt:.2f}, {max_r0_scalar_wt:.2f})")
mean_r0_scalar_pim_ls = np.mean(r0_scalar_pim_ls)
min_r0_scalar_pim_ls = np.min(r0_scalar_pim_ls)
max_r0_scalar_pim_ls = np.max(r0_scalar_pim_ls)
print(f"pim_lr: {mean_r0_scalar_wt:.2f} ({min_r0_scalar_wt:.2f}, {max_r0_scalar_wt:.2f})")
# Plot maps of the original and rescaled values
plot_choropleths_with_histograms(shp, r0_scalar_wt, r0_scalar_pim_ls, show=True, save_path="results/plot_r0_scalars_nigeria.png")


# ----- Version just for all of Africa -----

regions = ["AFRO", "EMRO"]

# Geography
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
shp = gpd.read_file(filename="data/shp_africa_low_res.gpkg", layer="adm2")
shp = shp[shp["dot_name"].isin(dot_names)]
shp.set_index("dot_name", inplace=True)
shp = shp.loc[dot_names].reset_index()
# Load the curated data
df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
df_comp = df_comp[df_comp["year"] == start_year]
# Extract the data
wt = df_comp.set_index("dot_name").loc[dot_names, "prop_underwt"].values
pim = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values
# Convert to R0 modifiers
r0_scalar_wt = (1 / (1 + np.exp(24 * (0.22 - wt)))) + 0.2  # The 0.22 is the mean of Nigeria underwt
mean_r0_scalar_wt = np.mean(r0_scalar_wt)
min_r0_scalar_wt = np.min(r0_scalar_wt)
max_r0_scalar_wt = np.max(r0_scalar_wt)
# Option 1: Linear Rescaling (Min-Max Match)
pim_scaled = (pim - nig_min) / (nig_max - nig_min)
# pim_scaled = (pim - pim.min()) / (pim.max() - pim.min())  # Rescale to [0, 1]
r0_scalar_pim_ls = pim_scaled * (r0_scalar_wt.max() - r0_scalar_wt.min()) + r0_scalar_wt.min()
# Compute the mean, min, and max of the rescaled values
print(f"r0_scalar_wt: {mean_r0_scalar_wt:.2f} ({min_r0_scalar_wt:.2f}, {max_r0_scalar_wt:.2f})")
mean_r0_scalar_pim_ls = np.mean(r0_scalar_pim_ls)
min_r0_scalar_pim_ls = np.min(r0_scalar_pim_ls)
max_r0_scalar_pim_ls = np.max(r0_scalar_pim_ls)
print(f"pim_lr: {mean_r0_scalar_wt:.2f} ({min_r0_scalar_wt:.2f}, {max_r0_scalar_wt:.2f})")
# Plot maps of the original and rescaled values
plot_choropleths_with_histograms(shp, r0_scalar_wt, r0_scalar_pim_ls, show=True, save_path="results/plot_r0_scalars_africa.png")


# ----- Version of all of Africa, but zoom in on Nigeria -----

# Filter to only Nigeria
nigeria_mask = shp["adm0_name"] == "NIGERIA"
shp_nigeria = shp[nigeria_mask].copy()

# Use the mask to filter scalars — ensure correct alignment
r0_scalar_wt_nigeria = r0_scalar_wt[nigeria_mask.values]
r0_scalar_pim_ls_nigeria = r0_scalar_pim_ls[nigeria_mask.values]

# Plot only Nigeria
plot_choropleths_with_histograms(
    shp_nigeria, r0_scalar_wt_nigeria, r0_scalar_pim_ls_nigeria, show=True, save_path="results/plot_r0_scalars_africa_zoom_nigeria.png"
)
