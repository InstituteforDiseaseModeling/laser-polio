import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import sciris as sc
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import laser_polio as lp


def save_study_results(study, output_dir: Path, csv_name: str = "trials.csv"):
    """
    Saves the essential outputs of an Optuna study (best params, metadata,
    trials data) into the given directory. Caller is responsible for loading
    the study and doing anything else (like copying model configs).

    :param study: An already-loaded Optuna study object.
    :param output_dir: Where to write all output files.
    :param csv_name: Optional CSV filename for trial data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print a brief best-trial summary
    best = study.best_trial
    sc.printcyan("\nBest Trial:")
    print(f"  Value: {best.value}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save trials dataframe
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    df.to_csv(output_dir / csv_name, index=False)

    # Save best params
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=4)

    # Save metadata
    metadata = dict(study.user_attrs)  # copy user_attrs
    metadata["timestamp"] = metadata.get("timestamp") or datetime.now().isoformat()  # noqa: DTZ005
    metadata["study_name"] = study.study_name
    metadata["storage_url"] = study.storage_url
    try:
        metadata["laser_polio_git_info"] = sc.gitinfo()
    except Exception:
        metadata["laser_polio_git_info"] = "Unavailable (no .git info in Docker)"
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Study results saved to '{output_dir}'")


def plot_optuna(study_name, storage_url, output_dir=None):
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Default output directory to current working dir if not provided
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving Optuna plots to: {output_dir.resolve()}")

    # Optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.update_yaxes(type="log")
    fig1.write_html(output_dir / "plot_opt_history.html")

    # # Param importances - WARNING! Can be slow for large studies
    # try:
    #     fig2 = vis.plot_param_importances(study)
    #     fig2.write_html(output_dir / "plot_param_importances.html")
    # except Exception as ex:
    #     print("[WARN] Could not plot param importances:", ex)

    # Slice plots
    params = study.best_params.keys()
    for param in params:  # or study.search_space.keys()
        fig3 = vis.plot_slice(study, params=[param])
        # Set log scale on y-axis
        fig3.update_yaxes(type="log")
        # fig.update_layout(width=plot_width)
        fig3.write_html(output_dir / f"plot_slice_{param}.html")

    # Contour plots - WARNING! Can be slow for large studies
    # try:
    #     fig4 = vis.plot_contour(study, params=["r0", "radiation_k"])
    #     fig4.write_html(output_dir / "plot_contour_r0_radiation_k.html")
    # try:
    #     fig4 = vis.plot_contour(study, params=["r0", "gravity_k_exponent"])
    #     fig4.write_html(output_dir / "plot_contour_gravity_k_exponent.html")
    #     fig4 = vis.plot_contour(study, params=["r0", "gravity_c"])
    #     fig4.write_html(output_dir / "plot_contour_r0_gravity_c.html")
    # Candidate pairs to try
    # param_pairs = [
    #     ("r0", "radiation_k"),
    #     ("r0", "gravity_k_exponent"),
    #     ("r0", "gravity_c"),
    #     ("gravity_k_exponent", "gravity_c"),
    # ]
    # # Get set of all parameters in the study
    # all_params = {k for t in study.trials if t.params for k in t.params.keys()}
    # # Loop over param pairs and plot only if both exist
    # for x, y in param_pairs:
    #     if x in all_params and y in all_params:
    #         try:
    #             fig = vis.plot_contour(study, params=[x, y])
    #             fig.write_html(output_dir / f"plot_contour_{x}_{y}.html")
    #         except Exception as e:
    #             print(f"[WARN] Failed to plot {x} vs {y}: {e}")
    #     else:
    #         print(f"[SKIP] Missing one or both params: {x}, {y}")
    # print("done with countour plots")


def plot_case_diff_choropleth(shp, node_lookup, actual_cases, pred_cases, output_path, title="Case Count Difference"):
    """
    Plot a choropleth map showing the difference between actual and predicted case counts.

    Args:
        shp (GeoDataFrame): The shapefile GeoDataFrame
        node_lookup (dict): Dictionary mapping dot_names to administrative regions
        actual_cases (dict): Dictionary of actual case counts by region
        pred_cases (dict): Dictionary of predicted case counts by region
        output_path (Path): Path to save the plot
        title (str): Title for the plot
    """

    # Calculate differences
    regions = set(actual_cases.keys()) | set(pred_cases.keys())
    differences = {region: actual_cases.get(region, 0) - pred_cases.get(region, 0) for region in regions}

    # Create a copy of the shapefile and add the differences
    shp_copy = shp.copy()

    # Map the differences using adm01_name
    shp_copy["case_diff"] = shp_copy["adm01_name"].map(differences)

    # Create diverging colormap centered at 0
    max_abs_diff = max(abs(min(differences.values())), abs(max(differences.values())))
    vmin, vmax = -max_abs_diff, max_abs_diff

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # Plot choropleth
    ax_map = fig.add_subplot(gs[0])

    # Create mappable for custom colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Plot the map
    shp_copy.plot(
        column="case_diff",
        ax=ax_map,
        cmap="RdBu",  # Red-Blue diverging colormap
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar with custom labels
    cbar = plt.colorbar(sm, ax=ax_map)
    cbar.ax.text(0.5, 1.05, "Obs > pred", ha="center", va="bottom", transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "Obs < pred", ha="center", va="top", transform=cbar.ax.transAxes)

    ax_map.set_title(title)
    ax_map.axis("off")

    # Plot histogram
    ax_hist = fig.add_subplot(gs[1])
    ax_hist.hist(list(differences.values()), bins=20, color="gray", edgecolor="black")
    ax_hist.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax_hist.set_xlabel("Case Count Difference (Actual - Predicted)")
    ax_hist.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def get_shapefile_from_config(model_config):
    """
    Generate a shapefile from the model configuration.

    Args:
        model_config (dict): Model configuration dictionary containing region information

    Returns:
        tuple: (GeoDataFrame, dict) The generated shapefile and node lookup dictionary mapping dot_names to adm01 names
    """
    # Extract region information from config
    regions = model_config.get("regions", [])
    if not regions:
        raise ValueError("No regions specified in model config")

    # Get dot names for the regions
    dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

    # Load node lookup and create dot_name to adm01 mapping
    node_lookup_full = lp.get_node_lookup("data/node_lookup.json", dot_names)
    # Convert from index-based to dot_name-based dictionary
    node_lookup = {entry["dot_name"]: entry["adm01"] for entry in node_lookup_full.values()}

    # Load and filter shapefile
    shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")
    shp = shp[shp["dot_name"].isin(dot_names)]

    # Ensure correct ordering if needed
    shp = shp.set_index("dot_name").loc[dot_names].reset_index()

    # Add administrative region names from node_lookup
    shp["adm01_name"] = shp["dot_name"].map(node_lookup)

    # Verify we have all mappings
    missing_mappings = shp[shp["adm01_name"].isna()]["dot_name"].tolist()
    if missing_mappings:
        print(f"[WARN] Missing node_lookup mappings for: {missing_mappings}")

    return shp, node_lookup


def plot_targets(study, output_dir=None, shp=None):
    best = study.best_trial
    actual = best.user_attrs["actual"]
    preds = best.user_attrs["predicted"]

    # Load metadata and model config
    metadata_path = Path(output_dir) / "study_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"study_metadata.json not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_config = metadata.get("model_config", {})

    # Generate shapefile if not provided
    if shp is None and "adm01_cases" in actual:
        try:
            shp, node_lookup = get_shapefile_from_config(model_config)
            print("[INFO] Generated shapefile from model config")
        except Exception as e:
            print(f"[WARN] Could not generate shapefile: {e}")
            shp = None

    # Load region group labels from study_metadata.json
    region_groups = model_config.get("summary_config", {}).get("region_groups", {})
    region_labels = list(region_groups.keys())

    # Define consistent colors
    n_reps = len(preds)
    labels = ["Actual"] + [f"Rep {i + 1}" for i in range(n_reps)]
    cmap = cm.get_cmap("Dark2")  # tab10
    color_map = {label: cmap(i) for i, label in enumerate(labels)}

    # Total Infected
    plt.figure()
    plt.title("Total Infected")
    width = 0.2
    x = np.arange(len(labels))
    values = [actual["total_infected"][0]] + [rep["total_infected"][0] for rep in preds]
    plt.bar(x, values, width=width, color=[color_map[label] for label in labels])
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Cases")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_total_infected_comparison.png")

    # Yearly Cases
    years = list(range(2018, 2018 + len(actual["yearly_cases"])))
    plt.figure()
    plt.title("Yearly Cases")
    plt.plot(years, actual["yearly_cases"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(years, rep["yearly_cases"], "o-", label=label, color=color_map[label])
    plt.xlabel("Year")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_yearly_cases_comparison.png")
    # plt.show()

    # Monthly Cases
    months = list(range(1, 1 + len(actual["monthly_cases"])))
    plt.figure()
    plt.title("Monthly Cases")
    plt.plot(months, actual["monthly_cases"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(months, rep["monthly_cases"], "o-", label=label, color=color_map[label])
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_monthly_cases_comparison.png")
    # plt.show()

    # Monthly Timeseries Cases
    n_months = len(actual["monthly_timeseries"])
    months_series = pd.date_range(start="2018-01-01", periods=n_months, freq="MS")
    plt.figure()
    plt.title("Monthly Timeseries")
    plt.plot(months_series, actual["monthly_timeseries"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(months_series, rep["monthly_timeseries"], "o-", label=label, color=color_map[label])
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_monthly_timeseries_comparison.png")
    # plt.show()

    # adm0_cases (bar plot)
    adm0_actual = actual.get("adm0_cases")
    if adm0_actual:
        adm_labels = sorted(actual["adm0_cases"].keys())
        x = np.arange(len(adm_labels))
        # Get actual values
        actual_vals = [actual["adm0_cases"].get(adm, 0) for adm in adm_labels]
        plt.figure(figsize=(10, 6))
        plt.title("ADM0 Cases")
        # Plot actual as outlined bar
        plt.bar(x, actual_vals, width=0.6, edgecolor=color_map["Actual"], facecolor="none", linewidth=1.5, label="Actual")
        # Plot predicted reps as colored dots
        for i, rep in enumerate(preds):
            label = f"Rep {i + 1}"
            rep_vals = [rep["adm0_cases"].get(adm, 0) for adm in adm_labels]
            plt.scatter(
                x,
                rep_vals,
                label=label,
                color=color_map[label],  # your original style
                marker="o",
                s=50,
            )
        # Axis formatting
        plt.xticks(x, adm_labels, rotation=45, ha="right")
        plt.ylabel("Cases")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "plot_adm0_cases.png")

    # adm01_cases (bar plot)
    adm01_actual = actual.get("adm01_cases")
    if adm01_actual:
        adm_labels = sorted(actual["adm01_cases"].keys())
        x = np.arange(len(adm_labels))
        # Get actual values
        actual_vals = [actual["adm01_cases"].get(adm, 0) for adm in adm_labels]
        plt.figure(figsize=(10, 6))
        plt.title("ADM01 Regional Cases")
        # Plot actual as outlined bar
        plt.bar(x, actual_vals, width=0.6, edgecolor=color_map["Actual"], facecolor="none", linewidth=1.5, label="Actual")
        # Plot predicted reps as colored dots
        for i, rep in enumerate(preds):
            label = f"Rep {i + 1}"
            rep_vals = [rep["adm01_cases"].get(adm, 0) for adm in adm_labels]
            plt.scatter(
                x,
                rep_vals,
                label=label,
                color=color_map[label],  # your original style
                marker="o",
                s=50,
            )
        # Axis formatting
        plt.xticks(x, adm_labels, rotation=45, ha="right")
        plt.ylabel("Cases")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "plot_adm01_cases.png")

    # Plot choropleth of case count differences for each replicate
    if shp is not None and "adm01_cases" in actual:
        for i, rep in enumerate(preds):
            if "adm01_cases" in rep:
                plot_case_diff_choropleth(
                    shp=shp,
                    node_lookup=node_lookup,
                    actual_cases=actual["adm01_cases"],
                    pred_cases=rep["adm01_cases"],
                    output_path=output_dir / f"plot_case_diff_choropleth_rep{i + 1}.png",
                    title=f"Case Count Difference (Actual - Predicted) - Rep {i + 1}",
                )

    # Regional Cases (bar plot)
    x = np.arange(len(region_labels))
    width = 0.1
    plt.figure()
    plt.title("Regional Cases")
    plt.bar(x, actual["regional_cases"], width, label="Actual", color=color_map["Actual"])
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.bar(x + (i + 1) * width, rep["regional_cases"], width, label=f"Rep {i + 1}", color=color_map[label])
    plt.xticks(x + width * (len(preds) // 2), region_labels)
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_regional_cases_comparison.png")
    # plt.show()

    # Total Nodes with Cases
    plt.figure()
    plt.title("Total Nodes with Cases")
    width = 0.2
    x = np.arange(1 + len(preds))
    values = [actual["nodes_with_cases_total"][0]] + [rep["nodes_with_cases_total"][0] for rep in preds]
    labels = ["Actual"] + [f"Rep {i + 1}" for i in range(len(preds))]
    plt.bar(x, values, width=width, color=[color_map[lbl] for lbl in labels])
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Nodes")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_nodes_with_cases_total.png")

    # Monthly Nodes with Cases
    n_months = len(actual["nodes_with_cases_timeseries"])
    months = list(range(1, n_months + 1))
    plt.figure()
    plt.title("Monthly Nodes with Cases")
    plt.plot(months, actual["nodes_with_cases_timeseries"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(months, rep["nodes_with_cases_timeseries"], "o-", label=f"Rep {i + 1}", color=color_map[label])
    plt.xlabel("Month")
    plt.ylabel("Number of Nodes with ≥1 Case")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_nodes_with_cases_timeseries.png")

    # Plot likelihoods

    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import Normalize
    # from matplotlib.cm import ScalarMappable
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # # Plot monthly timeseries with annual maps
    # if shp is not None:
    #     months_series
    #     years = sorted(set(months_series.year))

    #     # Compute yearly sum per node
    #     months_df = pd.DataFrame(monthly_cases_per_node, index=months_series)
    #     annual_cases_by_node = {year: months_df.loc[str(year)].sum(axis=0).values for year in years}

    #     # Normalize color scale across years
    #     global_min = min(np.min(v) for v in annual_cases_by_node.values())
    #     global_max = max(np.max(v) for v in annual_cases_by_node.values())
    #     norm = Normalize(vmin=global_min, vmax=global_max)
    #     cmap = truncate_colormap(cmap_name, minval=0.1, maxval=0.9)

    #     # Create figure
    #     fig, ax = plt.subplots(figsize=(16, 6))
    #     ax.plot(months_series, actual_timeseries, "o-", label="Actual", linewidth=2)
    #     ax.set_title("Monthly Timeseries with Annual Maps")
    #     ax.set_xlabel("Month")
    #     ax.set_ylabel("Cases")
    #     ax.legend()

    #     # Plot inset maps aligned under each year
    #     for i, year in enumerate(years):
    #         center_date = pd.Timestamp(f"{year}-07-01")  # middle of the year
    #         x_pos = center_date

    #         # Convert data coords to figure fraction
    #         trans = ax.transData.transform((x_pos.toordinal(), ax.get_ylim()[0]))
    #         inv = fig.transFigure.inverted().transform(trans)
    #         x_fig, y_fig = inv

    #         # Add inset axes
    #         axins = inset_axes(
    #             ax,
    #             width="5%", height="20%",  # relative to figure
    #             bbox_to_anchor=(x_fig - 0.025, 0.02, 0.05, 0.15),
    #             bbox_transform=fig.transFigure,
    #             loc="lower left",
    #             borderpad=0,
    #         )

    #         shp["cases"] = annual_cases_by_node[year]
    #         shp.plot(
    #             column="cases",
    #             ax=axins,
    #             cmap=cmap,
    #             norm=norm,
    #             edgecolor="white",
    #             linewidth=0.1,
    #             legend=False
    #         )
    #         axins.set_title(f"{year}", fontsize=8)
    #         axins.set_axis_off()

    #     # Shared colorbar
    #     sm = ScalarMappable(cmap=cmap, norm=norm)
    #     sm._A = []
    #     cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.01)
    #     cbar.set_label("Annual Node-Level Cases")

    #     plt.tight_layout()
    #     plt.show()


def plot_likelihoods(study, output_dir=None, use_log=True):
    best = study.best_trial
    likelihoods = best.user_attrs["likelihoods"]
    exclude_keys = {"total_log_likelihood"}
    keys = [k for k in likelihoods if k not in exclude_keys]
    values = [likelihoods[k] for k in keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(keys, values)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel("Log Likelihood")
    else:
        ax.set_ylabel("Likelihood")
    ax.set_title("Calibration Log-Likelihoods by Component")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom" if not use_log else "top",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "plot_likelihoods.png")
    # plt.show()


def plot_runtimes(study, output_dir=None):
    # Collect runtimes of completed trials
    durations = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.datetime_start and trial.datetime_complete:
            duration = trial.datetime_complete - trial.datetime_start
            durations.append(duration.total_seconds() / 60)  # Convert to minutes

    # Plot histogram with mean runtime
    if durations:
        avg_runtime = sum(durations) / len(durations)

        plt.figure(figsize=(8, 5))
        plt.hist(durations, bins=20, edgecolor="black", alpha=0.75)

        # Add vertical dashed mean line
        plt.axvline(avg_runtime, color="red", linestyle="--", linewidth=2, label=f"Mean = {avg_runtime:.2f} min")

        # Annotated title
        plt.title("Histogram of Optuna Trial Runtimes")
        plt.xlabel("Trial Runtime (minutes)")
        plt.ylabel("Number of Trials")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_runtimes.png")
        # plt.show()
    else:
        print("No completed trials with valid timestamps to plot.")


def load_region_group_labels(model_config_path):
    with open(model_config_path) as f:
        config = yaml.safe_load(f)
    region_groups = config.get("summary_config", {}).get("region_groups", {})
    return list(region_groups.keys())
