import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors

import laser_polio as lp

# --- Setup scope ---
regions = ["NIGERIA"]
start_year = 2018
n_days = 365 * 6

# --- Load the shapefile ---
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
node_lookup = lp.get_node_lookup("data/node_lookup.json", dot_names)
shp = gpd.read_file(filename="data/shp_africa_low_res.gpkg", layer="adm2")
shp = shp[shp["dot_name"].isin(dot_names)]
shp.set_index("dot_name", inplace=True)
shp = shp.loc[dot_names].reset_index()

# --- Load epidemiological data ---
epi = lp.get_epi_data("data/epi_africa_20250421.h5", dot_names, node_lookup, start_year, n_days)

# --- Convert time to quarter ---
epi["quarter"] = epi["date"].dt.to_period("Q")

# --- Count infections by quarter and location ---
quarterly_cases = epi.groupby(["quarter", "dot_name"])["cases"].sum().reset_index(name="case_count")

# # --- Join data and plot maps ---
# for q, shp_q in quarterly_cases.groupby("quarter"):
#     merged = shp.merge(shp_q, on="dot_name", how="left")
#     ax = merged.plot(
#         column="case_count",
#         cmap="viridis",
#         edgecolor="black",
#         legend=True,
#         figsize=(10, 8),
#         missing_kwds={"color": "lightgrey", "label": "No data"},
#     )
#     ax.set_title(f"Nigeria Case Counts - {q}", fontsize=14)
#     ax.axis("off")
#     plt.tight_layout()
#     plt.show()

# Prepare data
quarters = sorted(quarterly_cases["quarter"].unique())
n = len(quarters)
cols = 4
rows = 6

# Create figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.flatten()

# Custom colormap with grey for 0
cmap = plt.cm.rainbow
cmap = cmap.copy()  # avoid modifying global cmap
cmap.set_under("lightgrey")

# Normalize: use consistent scale across all plots
global_max = quarterly_cases["case_count"].max()
norm = colors.Normalize(vmin=0.1, vmax=global_max)

# Generate plots
for i, q in enumerate(quarters):
    ax = axes[i]
    shp_q = quarterly_cases[quarterly_cases["quarter"] == q]
    merged = shp.merge(shp_q, on="dot_name", how="left")
    merged["case_count"] = merged["case_count"].fillna(0)

    merged.plot(
        column="case_count",
        cmap=cmap,
        norm=norm,
        # edgecolor="black",
        linewidth=0.3,
        ax=ax,
        legend=False,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_title(f"{q}", fontsize=10)
    ax.axis("off")

# Turn off any unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Shared colorbar below the plots

# Adjust layout to leave space on the right
plt.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)

# Add colorbar on the right
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
fig.colorbar(sm, cax=cbar_ax, orientation="vertical", label="Case Count")

# Add main title
plt.suptitle("Nigeria Case Counts by Quarter", fontsize=16)
plt.savefig("results/actual_case_counts_by_quarter_nigeria.png", bbox_inches="tight", dpi=300)
plt.show()
