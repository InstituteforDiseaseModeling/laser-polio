from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

__all__ = [
    "plot_choropleth_and_hist",
    "plot_init_immun_grid",
    "plot_pars",
]


def plot_pars(pars, shp, results_path):
    """
    Plot parameters on a map.

    Args:
        pars (dict): Dictionary of parameters to plot.
        results_path (Path or str): Path to save the plots.
    """

    # Create the results directory if it doesn't exist
    plot_path = results_path / "pars_plots"
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    # Maps: n_ppl, cbr, init_prev, r0_scalars
    pars_to_map = ["n_ppl", "cbr", "init_prev", "r0_scalars", "vx_prob_sia", "vx_prob_ri", "vx_prob_ipv"]
    for par in pars_to_map:
        if par is not None:
            try:
                values = pars[par]
                plot_choropleth_and_hist(shp, par, values, plot_path)
            except Exception:
                print(f"\n❌ Could not plot par: {par}")

    # Custom maps: init_immun, sia_schedule, seed_schedule
    if "init_immun" in pars:
        plot_init_immun_grid(shp, pars["init_immun"], plot_path)
    if "sia_schedule" in pars:
        plot_sia_schedule(shp, pars["sia_schedule"], plot_path)
    if "seed_schedule" in pars:
        plot_seed_schedule(shp, pars["seed_schedule"], pars["node_lookup"], plot_path)

    # TODO: Other: age_pyramid_path, vx_prob_ri


def plot_choropleth_and_hist(shp, par, values, results_path, cmap="viridis", figsize=(8, 8)):
    """
    Plot a choropleth map with a histogram underneath.

    Args:
        shp (GeoDataFrame): The shapefile GeoDataFrame.
        par (str): Name of the parameter.
        values (array): Values to plot. Must match len(shp).
        results_path (Path or str): Path to save the plot.
        cmap (str): Matplotlib colormap.
        figsize (tuple): Size of the figure.
    """
    shp_copy = shp.copy()
    shp_copy[par] = values

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_map = fig.add_subplot(gs[0])
    shp_copy.plot(column=par, ax=ax_map, legend=True, cmap=cmap)
    ax_map.set_title(par)
    ax_map.axis("off")

    ax_hist = fig.add_subplot(gs[1])
    ax_hist.hist(values, bins=20, color="gray", edgecolor="black")
    # ax_hist.set_title(f"{par} distribution")
    ax_hist.set_xlabel(par)
    ax_hist.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(results_path / f"plot_{par}.png")
    plt.close(fig)


def plot_init_immun_grid(shp, init_immun_df, results_path, cmap="viridis", n_cols=4, figsize=(16, 12)):
    """
    Plot a grid of choropleth maps for all 'immunity_' columns in init_immun_df.

    Args:
        shp (GeoDataFrame): Shapefile GeoDataFrame.
        init_immun_df (DataFrame): DataFrame with columns like 'immunity_0_5', etc.
        results_path (Path): Path to save the output image.
        cmap (str): Colormap for choropleths.
        n_cols (int): Number of columns in the grid layout.
        figsize (tuple): Overall figure size.
    """

    # Filter 'immunity_' columns
    immunity_cols = [col for col in init_immun_df.columns if col.startswith("immunity_")]
    n_plots = len(immunity_cols)
    n_rows = int(np.ceil(n_plots / n_cols))

    # Compute global color scale
    vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []  # Hack to allow colorbar for mappable without plot

    # Set up grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(immunity_cols):
        shp_copy = shp.copy()
        shp_copy[col] = init_immun_df[col].values  # Align by index

        shp_copy.plot(
            column=col,
            ax=axes[i],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,  # Shared scale
        )
        axes[i].set_title(col, fontsize=10)
        axes[i].axis("off")

    # Turn off any extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Immunity")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for colorbar
    plt.savefig(results_path / "plot_init_immun.png")
    plt.close(fig)


def plot_sia_schedule(shp, sia_schedule, results_path, n_cols=6, figsize=(20, 12)):
    """
    Plot all SIA rounds in a grid with covered areas in blue and others in grey.

    Args:
        shp (GeoDataFrame): Must be indexed so that row i corresponds to node i.
        sia_schedule (list of dict): Each dict contains 'date', 'age_range', 'vaccinetype', 'nodes'.
        results_path (Path): Path to save the full grid figure.
        n_cols (int): Number of columns in the grid.
        figsize (tuple): Overall figure size.
    """
    n_sias = len(sia_schedule)
    if n_sias <= n_cols:
        n_rows, n_cols = 1, n_sias
    else:
        n_rows = int(np.ceil(n_sias / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    for i, sia in enumerate(sia_schedule):
        shp_copy = shp.copy()
        shp_copy["covered"] = False
        shp_copy.loc[sia["nodes"], "covered"] = True
        shp_copy["covered"] = pd.Categorical(shp_copy["covered"], categories=[False, True])

        ax = axes[i]
        # Use categorical coloring for True/False
        cmap = ListedColormap(["lightgrey", "blue"])
        shp_copy.plot(
            column="covered",
            ax=ax,
            # edgecolor="black",
            cmap=cmap,
            legend=False,
        )

        age_lo = int(sia["age_range"][0] / 365)
        age_hi = int(sia["age_range"][1] / 365)
        date_str = sia["date"].strftime("%Y-%m-%d")
        title = f"{date_str}\n{age_lo}-{age_hi} yrs, {sia['vaccinetype']}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add shared legend
    covered_patch = mpatches.Patch(color="blue", label="Covered")
    uncovered_patch = mpatches.Patch(color="lightgrey", label="Not covered")
    fig.legend(handles=[uncovered_patch, covered_patch], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig(results_path / "plot_sia_schedule.png")
    plt.close(fig)


def plot_seed_schedule(shp, seed_schedule, node_lookup, results_path, n_cols=4, figsize=(16, 12), cmap="Oranges"):
    """
    Plot a grid of seed event maps showing where and when seeding occurs, colored by prevalence.

    Args:
        shp (GeoDataFrame): Shapefile with rows corresponding to node ids.
        seed_schedule (list of dict): List of seed events, each with 'date', 'dot_name', 'prevalence'.
        node_lookup (dict): Mapping from node_id to dict with 'dot_name'.
        results_path (Path): Directory to save the output figure.
        n_cols (int): Number of columns in the plot grid.
        figsize (tuple): Size of the full figure.
        cmap (str): Colormap for prevalence.
    """
    # Map dot_names to node_ids
    dotname_to_node = {v["dot_name"]: k for k, v in node_lookup.items()}

    # Auto-compute optimal rows/cols if small number of events
    n_events = len(seed_schedule)
    if n_events <= n_cols:
        n_rows, n_cols = 1, n_events
    else:
        n_rows = int(np.ceil(n_events / n_cols))

    prevalences = [event["prevalence"] for event in seed_schedule]
    max_prevalence = max(prevalences) if prevalences else 1
    norm = Normalize(vmin=0, vmax=max_prevalence)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, event in enumerate(seed_schedule):
        ax = axes[i]
        node_id = dotname_to_node.get(event["dot_name"], None)
        if node_id is None:
            print(f"Warning: dot_name '{event['dot_name']}' not found in node_lookup")
            shp.plot(ax=ax, color="lightgrey", edgecolor="black")
            ax.set_title(f"{event['date']}\n{event['dot_name']}\n(prevalence N/A)", fontsize=9)
            ax.axis("off")
            continue

        shp_copy = shp.copy()
        shp_copy["prevalence"] = 0
        shp_copy.at[node_id, "prevalence"] = event["prevalence"]

        # Plot: base in grey, seeded in color
        shp_copy.plot(ax=ax, color="lightgrey", edgecolor="black")
        shp_copy[shp_copy["prevalence"] > 0].plot(ax=ax, column="prevalence", cmap=cmap, norm=norm, edgecolor="black")

        ax.set_title(f"{event['date']}\n{event['dot_name']}\n(prevalence {event['prevalence']})", fontsize=8)
        ax.axis("off")

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Prevalence")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(results_path / "seed_schedule_grid.png")
    plt.close(fig)


# def plot_choropleth(shp, par, values, results_path, cmap="viridis", figsize=(8, 6)):
#     """
#     Plot a separate choropleth map for each parameter.

#     Args:
#         shp (GeoDataFrame): The shapefile GeoDataFrame.
#         par (str): Name of par
#         values (array): Values to plot. Must match len(shp).
#         results_path (str): Path to save the plots.
#         cmap (str): Matplotlib colormap.
#         figsize (tuple): Size of each individual figure.
#     """
#     shp_copy = shp.copy()
#     shp_copy[par] = values

#     fig, ax = plt.subplots(figsize=figsize)
#     shp_copy.plot(column=par, ax=ax, legend=True, cmap=cmap)
#     ax.set_title(par)
#     ax.axis("off")
#     plt.tight_layout()
#     plt.savefig(results_path / f"plot_par_{par}.png")
#     # plt.show()
