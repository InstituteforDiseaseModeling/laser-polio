from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laser_core.demographics.pyramid import load_pyramid_csv
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

__all__ = [
    "plot_age_pyramid",
    "plot_choropleth_and_hist",
    "plot_cum_new_exposed_paralyzed",
    "plot_cum_ri_vx",
    "plot_cum_vx_sia",
    "plot_infected_by_node",
    "plot_infected_by_node_strain",
    "plot_infected_choropleth",
    "plot_infected_choropleth_by_strain",
    "plot_infected_dot_map",
    "plot_init_immun_grid",
    "plot_network",
    "plot_new_exposed_by_strain",
    "plot_node_pop",
    "plot_pars",
    "plot_total_seir_counts",
    "plot_vital_dynamics",
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

    # Maps: init_pop, cbr, init_prev, r0_scalars
    pars_to_map = ["init_pop", "cbr", "init_prev", "r0_scalars", "vx_prob_sia", "vx_prob_ri", "vx_prob_ipv"]
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

    fig = plt.figure(figsize=figsize, constrained_layout=True)
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
    fig.subplots_adjust(right=0.88)  # Leave space for colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Immunity")

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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
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
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Prevalence")

    plt.savefig(results_path / "seed_schedule_grid.png")
    plt.close(fig)


def plot_node_pop(results, nodes, save=False, results_path=None):
    """Plot population over time for each node."""
    plt.figure(figsize=(10, 6))
    for node in nodes:
        pop = results.pop[:, node]
        plt.plot(pop, label=f"Node {node}")
    plt.title("Node Population")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Population")
    plt.grid()
    if save:
        plt.savefig(results_path / "node_population.png")
    if not save:
        plt.show()


def plot_total_seir_counts(results, save=False, results_path=None):
    """Plot total SEIR counts over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.sum(results.S, axis=1), label="Susceptible (S)")
    plt.plot(np.sum(results.E, axis=1), label="Exposed (E)")
    plt.plot(np.sum(results.I, axis=1), label="Infectious (I)")
    plt.plot(np.sum(results.R, axis=1), label="Recovered (R)")
    plt.plot(np.sum(results.paralyzed, axis=1), label="Paralyzed")
    plt.title("SEIR Dynamics in Total Population")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(results_path / "total_seir_counts.png")
    if not save:
        plt.show()


def plot_cum_new_exposed_paralyzed(results, save=False, results_path=None):
    """Plot cumulative new exposed and paralyzed cases."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(np.sum(results.new_exposed, axis=1)), label="Cumulative Exposed")
    plt.plot(np.cumsum(np.sum(results.new_potentially_paralyzed, axis=1)), label="Cumulative Potentially Paralyzed")
    plt.plot(np.cumsum(np.sum(results.new_paralyzed, axis=1)), label="Cumulative Paralyzed")
    plt.title("Cumulative New Exposed, Potentially Paralyzed, and Paralyzed")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Cumulative count")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(results_path / "cumulative_new_exposed_potentially_paralyzed.png")
    if not save:
        plt.show()


def plot_infected_by_node(results, nodes, save=False, results_path=None):
    """Plot infected population by node over time."""
    plt.figure(figsize=(10, 6))
    for node in nodes:
        plt.plot(results.I[:, node], label=f"Node {node}")
    plt.title("Infected Population by Node")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(results_path / "n_infected_by_node.png")
    if not save:
        plt.show()


def plot_infected_by_node_strain(results, pars, save=False, results_path=None, figsize=(15, 20)):
    """Plot infected population by node for each strain, with a subplot for each strain."""
    # Get the strain-specific infection data
    I_by_strain = results.I_by_strain  # Shape: (time, nodes, strains)
    n_time, n_nodes, n_strains = I_by_strain.shape

    # Create reverse mapping from strain index to strain name
    strain_names = {v: k for k, v in pars.strain_ids.items()}

    # Set up subplots - stack vertically (one column)
    n_rows = n_strains
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    # Handle case where we have only one subplot
    if n_strains == 1:
        axes = [axes]
    else:
        # axes is already a 1D array when n_cols=1, but ensure it's iterable
        axes = axes if isinstance(axes, np.ndarray) else [axes]

    # Plot each strain
    for strain_idx in range(n_strains):
        ax = axes[strain_idx]
        strain_name = strain_names.get(strain_idx, f"Strain {strain_idx}")

        # Plot infection timeseries for each node for this strain
        for node_idx in range(n_nodes):
            # Get node label
            if pars.node_lookup and node_idx in pars.node_lookup:
                # Use the last part of dot_name for cleaner labels
                node_label = pars.node_lookup[node_idx].get("dot_name", f"Node {node_idx}").split(":")[-1]
            else:
                node_label = f"Node {node_idx}"

            # Plot this node's infections for this strain
            infections = I_by_strain[:, node_idx, strain_idx]

            # Only plot if there are any infections (to reduce clutter)
            if np.sum(infections) > 0:
                ax.plot(infections, label=node_label, alpha=0.7)

        # Formatting
        ax.set_title(f"{strain_name} Infections by Node", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Number of Infected")
        ax.grid(True, alpha=0.3)

        # Add text indicating no infections if no lines plotted
        if len(ax.get_lines()) == 0:
            ax.text(0.5, 0.5, "No infections", transform=ax.transAxes, ha="center", va="center", fontsize=12, alpha=0.5)

    # Turn off any unused subplots
    for idx in range(n_strains, len(axes)):
        axes[idx].axis("off")

    # Overall formatting
    plt.tight_layout()

    # Save or show
    if save:
        if results_path is None:
            raise ValueError("Please provide a results_path to save the plot.")
        plot_path = Path(results_path) / "infected_by_node_strain.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_new_exposed_by_strain(results, pars, save=False, results_path=None, figsize=(20, 20)):
    """
    Plot new exposures by strain in a 3x3 grid.
    Rows: VDPV2, Sabin2, nOPV2
    Columns: Total exposures, Transmission only, SIA only
    """
    # Check if we have the required exposure data
    if not hasattr(results, "new_exposed_by_strain"):
        print("No strain-specific exposure data available")
        return

    # Get the strain-specific exposure data
    new_exposed_by_strain = results.new_exposed_by_strain  # Shape: (time, nodes, strains)
    n_time, n_nodes, n_strains = new_exposed_by_strain.shape

    # Create reverse mapping from strain index to strain name
    strain_names = {v: k for k, v in pars.strain_ids.items()}

    # Set up 3x3 subplot grid
    n_rows = 3  # One for each strain
    n_cols = 3  # Total, Transmission, SIA

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)

    # Column titles
    col_titles = ["Total New Exposures", "Transmission Only", "SIA Only"]

    # Plot each strain (row)
    for strain_idx in range(min(n_strains, 3)):  # Limit to 3 strains
        strain_name = strain_names.get(strain_idx, f"Strain {strain_idx}")

        # Get total exposures for this strain (sum across all nodes)
        total_exposures = np.sum(new_exposed_by_strain[:, :, strain_idx], axis=1)

        # Calculate transmission and SIA exposures
        if hasattr(results, "sia_new_exposed_by_strain"):
            sia_exposures = np.sum(results.sia_new_exposed_by_strain[:, :, strain_idx], axis=1)
            trans_exposures = total_exposures - sia_exposures
        else:
            # If no SIA data, assume all exposures are from transmission
            trans_exposures = total_exposures
            sia_exposures = np.zeros_like(total_exposures)

        # Column 1: Total exposures
        ax = axes[strain_idx, 0]
        if np.any(total_exposures > 0):
            ax.plot(total_exposures, linewidth=2, color="black", label="Total")
        else:
            ax.text(0.5, 0.5, "No exposures", transform=ax.transAxes, ha="center", va="center", fontsize=10, alpha=0.5)

        ax.set_title(f"{strain_name}\n{col_titles[0]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("New Exposures per Day")
        ax.grid(True, alpha=0.3)

        # Column 2: Transmission only
        ax = axes[strain_idx, 1]
        if np.any(trans_exposures > 0):
            ax.plot(trans_exposures, linewidth=2, color="green", label="Transmission")
        else:
            ax.text(0.5, 0.5, "No transmission", transform=ax.transAxes, ha="center", va="center", fontsize=10, alpha=0.5)

        ax.set_title(f"{strain_name}\n{col_titles[1]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("New Exposures per Day")
        ax.grid(True, alpha=0.3)

        # Column 3: SIA only
        ax = axes[strain_idx, 2]
        if np.any(sia_exposures > 0):
            ax.plot(sia_exposures, linewidth=2, color="red", label="SIA")
        else:
            ax.text(0.5, 0.5, "No SIA", transform=ax.transAxes, ha="center", va="center", fontsize=10, alpha=0.5)

        ax.set_title(f"{strain_name}\n{col_titles[2]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("New Exposures per Day")
        ax.grid(True, alpha=0.3)

    # Set x-labels for bottom row
    for col in range(n_cols):
        axes[n_rows - 1, col].set_xlabel("Time (days)")

    # Overall formatting
    plt.suptitle("New Exposures by Strain and Source", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save or show
    if save:
        if results_path is None:
            raise ValueError("Please provide a results_path to save the plot.")
        plot_path = Path(results_path) / "new_exposed_by_strain_detailed.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_infected_dot_map(results, pars, nodes, save=False, results_path=None, n_panels=6):
    """Plot infected population as dots on a map."""
    rows, cols = 2, int(np.ceil(n_panels / 2))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), sharex=True, sharey=True, constrained_layout=True)
    axs = axs.ravel()  # Flatten in case of non-square grid
    timepoints = np.linspace(0, pars.dur, n_panels, dtype=int)
    lats = [pars.node_lookup[i]["lat"] for i in nodes]
    lons = [pars.node_lookup[i]["lon"] for i in nodes]
    # Scale population for plotting (adjust scale_factor as needed)
    scale_factor = 5  # tweak this number to look good visually
    sizes = np.array(pars.init_pop)
    sizes = np.log1p(sizes) * scale_factor
    # Get global min and max for consistent color scale
    infection_min = np.min(results.I)
    infection_max = np.max(results.I)
    for i, ax in enumerate(axs[:n_panels]):  # Ensure we don't go out of bounds
        t = timepoints[i]
        infection_counts = results.I[t, :]
        scatter = ax.scatter(
            lons, lats, c=infection_counts, s=sizes, cmap="RdYlBu_r", edgecolors=None, alpha=0.9, vmin=infection_min, vmax=infection_max
        )
        ax.set_title(f"Timepoint {t}")
        # Show labels only on the leftmost and bottom plots
        if i % cols == 0:
            ax.set_ylabel("Latitude")
        else:
            ax.set_yticklabels([])
        if i >= n_panels - cols:
            ax.set_xlabel("Longitude")
        else:
            ax.set_xticklabels([])
    # Add a single colorbar for all plots
    fig.colorbar(scatter, ax=axs, location="right", fraction=0.05, pad=0.05, label="Infection Count")
    fig.suptitle("Infected Population by Node", fontsize=16)
    if save:
        if results_path is None:
            raise ValueError("Please provide a results path to save the plots.")
        plt.savefig(f"{results_path}/infected_map.png")
    else:
        plt.show()


def plot_infected_choropleth(results, pars, save=False, results_path=None, n_panels=6):
    """Plot infected population as choropleth maps."""
    rows, cols = 2, int(np.ceil(n_panels / 2))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), constrained_layout=True)
    axs = axs.ravel()
    timepoints = np.linspace(0, pars.dur, n_panels, dtype=int)
    shp = pars.shp.copy()  # Don't mutate original GeoDataFrame

    # Get global min/max for consistent color scale across panels
    infection_min = np.min(results.I[results.I > 0]) if np.any(results.I > 0) else 0
    infection_max = np.max(results.I)
    alpha = 0.9

    # Use rainbow colormap and truncate if desired
    cmap = plt.cm.get_cmap("rainbow")
    norm = mcolors.Normalize(vmin=infection_min, vmax=infection_max)

    for i, ax in enumerate(axs[:n_panels]):
        t = timepoints[i]
        infection_counts = results.I[t, :]  # shape = (num_nodes,)
        shp["infected"] = infection_counts
        shp["infected_masked"] = shp["infected"].replace({0: np.nan})  # Mask out zeros

        shp.plot(
            column="infected_masked",
            ax=ax,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            linewidth=0.1,
            edgecolor="white",
            legend=False,
            missing_kwds={"color": "lightgrey", "label": "Zero infections"},
        )
        ax.set_title(f"Infections at t={t}")
        ax.set_axis_off()

    # Add a shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.03, pad=0.01)
    cbar.solids.set_alpha(alpha)
    cbar.set_label("Infection Count")
    fig.suptitle("Choropleth of Infected Population by Node", fontsize=16)

    if save:
        if results_path is None:
            raise ValueError("Please provide a results path to save the plots.")
        plt.savefig(results_path / "infected_choropleth.png")
    else:
        plt.show()


def plot_infected_choropleth_by_strain(results, pars, save=False, results_path=None, n_panels=6):
    """
    Plot separate choropleth figures for each strain using results.I_by_strain.
    Creates one figure per strain, each with n_panels showing infection counts over time.
    """
    timepoints = np.linspace(0, pars.dur, n_panels, dtype=int)
    shp = pars.shp.copy()  # Don't mutate original GeoDataFrame

    # Get strain information
    strain_ids = pars.strain_ids
    # results.I_by_strain has shape (time, nodes, strains)
    I_by_strain = results.I_by_strain
    for strain_idx, strain_id in enumerate(strain_ids):
        # Get data for this strain across all time and nodes
        strain_data = I_by_strain[:, :, strain_idx]  # shape: (time, nodes)

        # Get global min/max for consistent color scale across panels for this strain
        infection_min = np.min(strain_data[strain_data > 0]) if np.any(strain_data > 0) else 0
        infection_max = np.max(strain_data)

        # Skip strains with no infections
        if infection_max == 0:
            print(f"Skipping strain {strain_id} - no infections found")
            continue

        alpha = 0.9
        rows, cols = 2, int(np.ceil(n_panels / 2))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), constrained_layout=True)
        axs = axs.ravel()

        # Use rainbow colormap
        cmap = plt.cm.get_cmap("rainbow")
        norm = mcolors.Normalize(vmin=infection_min, vmax=infection_max)

        for i, ax in enumerate(axs[:n_panels]):
            t = timepoints[i]
            infection_counts = strain_data[t, :]  # shape = (num_nodes,)
            shp["infected"] = infection_counts
            shp["infected_masked"] = shp["infected"].replace({0: np.nan})  # Mask out zeros
            shp.plot(
                column="infected_masked",
                ax=ax,
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                linewidth=0.1,
                edgecolor="white",
                legend=False,
                missing_kwds={"color": "lightgrey", "label": "Zero infections"},
            )
            ax.set_title(f"Strain {strain_id} Infections at t={t}")
            ax.set_axis_off()

        # Add a shared colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.03, pad=0.01)
        cbar.solids.set_alpha(alpha)
        cbar.set_label("Infection Count")
        fig.suptitle(f"Choropleth of Infected Population by Node - Strain {strain_id}", fontsize=16)

        if save:
            if results_path is None:
                raise ValueError("Please provide a results path to save the plots.")
            plt.savefig(
                results_path / f"infected_choropleth_strain_{strain_id}.png", dpi=300, format="png", facecolor="white", edgecolor="none"
            )
            plt.close(fig)
        else:
            plt.show()


def plot_network(network, save=False, results_path=""):
    """
    Plot a heatmap of the network & a histogram of the proportions of infections leaving each node.
    """
    # Handle paths
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Convert to array
    network_array = np.array(network)

    # Mask zeros
    masked_network = np.ma.masked_where(network_array == 0.0, network_array)

    # Create custom colormap
    cmap = cm.get_cmap("plasma").copy()
    cmap.set_bad("white")

    # Plot heatmap using imshow
    im = axs[0].imshow(masked_network, cmap=cmap, origin="upper", interpolation="none")
    axs[0].set_title("Transmission Matrix (Heatmap)")
    axs[0].set_xlabel("Destination Node")
    axs[0].set_ylabel("Source Node")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    # Optionally annotate small networks
    if network_array.shape[0] <= 10:
        for i in range(network_array.shape[0]):
            for j in range(network_array.shape[1]):
                val = network_array[i, j]
                if val != 0.0:
                    axs[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    # Histogram
    values = network_array.sum(axis=1)
    axs[1].hist(values, bins=10, edgecolor="black", color="steelblue")
    axs[1].set_title("Proportion of Infections Leaving Each Node")
    axs[1].set_xlabel("Proportion")
    axs[1].set_ylabel("Count")
    axs[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # round x-axis
    axs[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # optional: round y-axis

    plt.tight_layout()

    if save:
        fig.savefig(results_path / "network.png", dpi=300)
    plt.close(fig)


def plot_age_pyramid(people, pars, sim, save=False, results_path=None):
    """Plot age pyramid comparison between expected and observed."""
    # Expected age distribution
    exp_ages = pd.read_csv(pars.age_pyramid_path)
    exp_ages["Total"] = exp_ages["M"] + exp_ages["F"]
    exp_ages["Proportion"] = exp_ages["Total"] / exp_ages["Total"].sum()

    # Observed age distribution
    obs_ages = ((people.date_of_birth[: people.count] * -1) + sim.t) / 365  # THIS IS WRONG
    pyramid = load_pyramid_csv(pars.age_pyramid_path)
    bins = pyramid[:, 0]
    # Add 105+ bin
    bins = np.append(bins, 105)
    age_bins = pd.cut(obs_ages, bins=bins, right=False)
    age_bins.value_counts().sort_index()
    obs_age_distribution = age_bins.value_counts().sort_index()
    obs_age_distribution = obs_age_distribution / obs_age_distribution.sum()  # Normalize

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = exp_ages["Age"]
    x = np.arange(len(x_labels))
    ax.plot(x, exp_ages["Proportion"], label="Expected", color="green", linestyle="-", marker="x")
    ax.plot(x, obs_age_distribution, label="Observed at end of sim", color="blue", linestyle="--", marker="o")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Proportion of Population")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_title("Age Distribution as Proportion of Total Population")
    ax.legend()  # Add legend
    plt.tight_layout()
    if save:
        plt.savefig(results_path / "age_distribution.png")
    if not save:
        plt.show()


def plot_vital_dynamics(results, save=False, results_path=None):
    """
    Plot births and deaths over time.
    This function originally plot births and deaths for each node, but we've switched it to be aggregated.
    This was because we weren't noticing errors with the node-wise plots and we don't have spatially
    varying inputs for fertility and mortality rates at this time.
    """
    # Calculate cumulative sums
    births_total = np.sum(results.births, axis=1)
    deaths_total = np.sum(results.deaths, axis=1)

    # Compute cumulative sums over time
    cum_births = np.cumsum(births_total)
    cum_deaths = np.cumsum(deaths_total)

    plt.figure(figsize=(10, 6))
    plt.plot(cum_births, label="Births", color="blue")
    plt.plot(cum_deaths, label="Deaths", color="red")
    plt.title("Cumulative births and deaths (All Nodes)")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(results_path / "cum_births_deaths.png")
    if not save:
        plt.show()


def plot_cum_ri_vx(results, save=False, results_path=None):
    """Plot cumulative RI vaccinated."""
    cum_ri_vaccinated = np.cumsum(results.ri_vaccinated, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(cum_ri_vaccinated)
    plt.title("Cumulative RI Vaccinated (includes efficacy)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Vaccinated")
    plt.grid()
    if save:
        plt.savefig(results_path / "cum_ri_vx.png")
    if not save:
        plt.show()


def plot_cum_vx_sia(results, save=False, results_path=None):
    """Plot cumulative SIA vaccinated."""
    cum_vx_sia = np.cumsum(results.sia_vaccinated, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(cum_vx_sia)
    plt.title("Supplemental Immunization Activity (SIA) Vaccination")
    plt.xlabel("Time (Timesteps)")
    plt.ylabel("Cumulative Vaccinated")
    plt.grid()
    if save:
        plt.savefig(results_path / "cum_sia_vx.png")
    if not save:
        plt.show()
