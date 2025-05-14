import matplotlib.pyplot as plt

__all__ = [
    "plot_choropleth_and_hist",
]


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
