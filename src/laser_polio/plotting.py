from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
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
    pars_to_map = ["n_ppl", "cbr", "init_prev", "r0_scalars"]
    for par in pars_to_map:
        values = pars[par]
        plot_choropleth_and_hist(shp, par, values, plot_path)

    # Map: init_immun
    plot_init_immun_grid(shp, pars["init_immun"], plot_path)

    # Map: sia_schedule
    plot_sia_schedule(shp, pars["sia_schedule"], plot_path)

    # TODO: Other: age_pyramid_path, vx_prob_ri, vx_prob_sia, seed_schedule


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
    all_vals = init_immun_df[immunity_cols].values.flatten()
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


def plot_sia_schedule(shp, sia_schedule, results_path, cmap="Reds", figsize=(8, 6)):
    """
    Plot a series of maps showing SIA coverage.

    Args:
        shp (GeoDataFrame): Shapefile GeoDataFrame, must be indexed by node number.
        sia_schedule (list of dict): Each dict must contain 'date', 'age_range', 'vaccinetype', 'nodes'.
        results_path (Path): Directory where plots will be saved.
        cmap (str): Colormap for covered areas.
        figsize (tuple): Size of each individual figure.
    """
    for i, sia in enumerate(sia_schedule):
        covered = [False] * len(shp)
        for node in sia["nodes"]:
            covered[node] = True

        shp_copy = shp.copy()
        shp_copy["covered"] = covered

        fig, ax = plt.subplots(figsize=figsize)
        shp_copy.plot(ax=ax, column="covered", cmap=cmap, legend=False, edgecolor="black")

        # Title with date, age range in years, and vaccine type
        age_lo = int(sia["age_range"][0] / 365)
        age_hi = int(sia["age_range"][1] / 365)
        date_str = sia["date"].strftime("%Y-%m-%d")
        title = f"SIA {i:02d}: {date_str}, ages {age_lo}-{age_hi} yrs, {sia['vaccinetype']}"
        ax.set_title(title, fontsize=12)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(results_path / f"sia_{i:02d}.png")
        plt.close(fig)


def plot_sia_schedule_grid(shp, sia_schedule, results_path, n_cols=4, figsize=(20, 12)):
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
    n_rows = int(np.ceil(n_sias / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, sia in enumerate(sia_schedule):
        shp_copy = shp.copy()
        shp_copy["covered"] = False
        shp_copy.loc[sia["nodes"], "covered"] = True

        ax = axes[i]
        # Use categorical coloring for True/False
        shp_copy.plot(
            column="covered",
            ax=ax,
            edgecolor="black",
            cmap="bwr",  # blue for True, red for False (or use ListedColormap)
            categorical=True,
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

    plt.tight_layout()
    plt.savefig(results_path / "sia_schedule_grid.png")
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
