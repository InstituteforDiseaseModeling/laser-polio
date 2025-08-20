import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Ensure we're using the Agg backend for better cross-platform compatibility
matplotlib.use("Agg")
import optuna
import optuna.visualization as vis
import pandas as pd
import sciris as sc
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    import cloud_calib_config as cfg
    from idmtools.assets import Asset
    from idmtools.assets import AssetCollection
    from idmtools.core.platform_factory import Platform
    from idmtools.entities import CommandLine
    from idmtools.entities.command_task import CommandTask
    from idmtools.entities.experiment import Experiment
    from idmtools.entities.simulation import Simulation
    from idmtools_platform_comps.utils.scheduling import add_schedule_config

    HAS_IDMTOOLS = True
except Exception:
    HAS_IDMTOOLS = False

import laser_polio as lp


def sweep_seed_best_comps(study, output_dir: Path = "results"):
    if not HAS_IDMTOOLS:
        raise ImportError("idmtools is not installed.")

    # Sort trials by best objective value (lower is better)
    top_trial = study.best_trial

    Platform("Idm", endpoint="https://comps.idmod.org", environment="CALCULON", type="COMPS")
    experiment = Experiment(name=f"laser-polio Best Trial from {study.study_name}", tags={"source": "optuna", "mode": "top-n"})

    for seed in range(10):
        overrides = top_trial.params.copy()
        overrides["save_plots"] = True
        overrides["seed"] = seed
        # You can include trial.number or trial.value as well

        # Write overrides file with trial-specific filename
        command = CommandLine(
            f"singularity exec --no-mount /app Assets/laser-polio_latest.sif "
            f"python3 -m laser_polio.run_sim "
            f"--model-config /app/calib/model_configs/{cfg.model_config} "
            f"--params-file overrides.json "
            # f"--init-pop-file=Assets/init_pop_nigeria_4y_2020_underwt_gravity_zinb_ipv.h5"
        )

        task = CommandTask(command=command)
        task.common_assets.add_assets(AssetCollection.from_id_file("calib/comps/laser.id"))
        # task.common_assets.add_directory("inputs")
        task.transient_assets.add_asset(Asset(filename="overrides.json", content=json.dumps(overrides)))

        # Wrap task in Simulation and add to experiment
        simulation = Simulation(task=task)
        simulation.tags.update({"description": "LASER-Polio"})  # , ".trial_rank": str(rank), ".trial_value": str(trial.value)})
        experiment.add_simulation(simulation)

        add_schedule_config(
            simulation, command=command, NumNodes=1, NumCores=12, NodeGroupName="idm_abcd", Environment={"NUMBA_NUM_THREADS": str(12)}
        )
    experiment.run(wait_until_done=True)
    exp_id_filepath = output_dir / "comps_exp.id"
    experiment.to_id_file(exp_id_filepath)


def run_top_n_on_comps(study, n=10, output_dir: Path = "results"):
    if not HAS_IDMTOOLS:
        raise ImportError("idmtools is not installed.")

    # Sort trials by best objective value (lower is better)
    top_trials = sorted([t for t in study.trials if t.state.name == "COMPLETE"], key=lambda t: t.value)[:n]

    Platform("Idm", endpoint="https://comps.idmod.org", environment="CALCULON", type="COMPS")
    experiment = Experiment(name=f"laser-polio top {n} from {study.study_name}", tags={"source": "optuna", "mode": "top-n"})

    for rank, trial in enumerate(top_trials, start=1):
        overrides = trial.params.copy()
        overrides["save_plots"] = True
        # You can include trial.number or trial.value as well

        # Write overrides file with trial-specific filename
        command = CommandLine(
            f"singularity exec --no-mount /app Assets/laser-polio_latest.sif "
            f"python3 -m laser_polio.run_sim "
            f"--model-config /app/calib/model_configs/{cfg.model_config} "
            f"--params-file overrides.json "
            # f"--init-pop-file=Assets/init_pop_nigeria_6y_2018_underwt_gravity_zinb_ipv.h5"
        )

        task = CommandTask(command=command)
        task.common_assets.add_assets(AssetCollection.from_id_file("calib/comps/laser.id"))
        # task.common_assets.add_directory("inputs")
        task.transient_assets.add_asset(Asset(filename="overrides.json", content=json.dumps(overrides)))

        # Wrap task in Simulation and add to experiment
        simulation = Simulation(task=task)
        simulation.tags.update({"description": "LASER-Polio", ".trial_rank": str(rank), ".trial_value": str(trial.value)})
        experiment.add_simulation(simulation)

        add_schedule_config(
            simulation, command=command, NumNodes=1, NumCores=12, NodeGroupName="idm_abcd", Environment={"NUMBA_NUM_THREADS": str(12)}
        )
    experiment.run(wait_until_done=True)
    exp_id_filepath = output_dir / "comps_exp.id"
    experiment.to_id_file(exp_id_filepath)


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
    output_dir = Path(output_dir) / "optuna_plots" if output_dir else Path.cwd()
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
    #     fig4 = vis.plot_contour(study, params=["r0", "radiation_k_log10"])
    #     fig4.write_html(output_dir / "plot_contour_r0_radiation_k.html")
    # try:
    #     fig4 = vis.plot_contour(study, params=["r0", "gravity_k_exponent"])
    #     fig4.write_html(output_dir / "plot_contour_gravity_k_exponent.html")
    #     fig4 = vis.plot_contour(study, params=["r0", "gravity_c"])
    #     fig4.write_html(output_dir / "plot_contour_r0_gravity_c.html")
    # Candidate pairs to try
    # param_pairs = [
    #     ("r0", "radiation_k_log10"),
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


def plot_case_diff_choropleth_temporal(
    shp, actual_cases_by_period, pred_cases_by_period, output_path, title="Case Count Difference by Period"
):
    """
    Plot choropleth maps showing the difference between actual and predicted case counts
    for multiple time periods using nested dictionary structure.

    Args:
        shp (GeoDataFrame): The shapefile GeoDataFrame with region-level geometries
        actual_cases_by_period (dict): Nested dictionary of actual case counts {region: {period: count}}
        pred_cases_by_period (dict): Nested dictionary of predicted case counts {region: {period: count}}
        output_path (Path): Path to save the plot
        title (str): Title for the plot
    """

    # Extract periods and regions from nested dictionary structure
    # Extract all unique periods from all regions
    all_periods = set()
    for region_dict in actual_cases_by_period.values():
        all_periods.update(region_dict.keys())
    for region_dict in pred_cases_by_period.values():
        all_periods.update(region_dict.keys())
    periods = sorted(all_periods)

    # Restructure data by period for easier processing
    period_actual = {}
    period_pred = {}

    for period in periods:
        period_actual[period] = {}
        period_pred[period] = {}

        # Extract data for this period from all regions
        for region, region_data in actual_cases_by_period.items():
            period_actual[period][region] = region_data.get(period, 0)

        for region, region_data in pred_cases_by_period.items():
            period_pred[period][region] = region_data.get(period, 0)

    # Use periods in the order they appear in the dictionary
    # Create figure with subplots
    n_periods = len(periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(8 * n_periods, 8))
    if n_periods == 1:
        axes = [axes]

    # Calculate global min/max for consistent color scale
    all_diffs = []
    period_diffs = {}

    # Calculate differences for each period
    for period in periods:
        actual_data = period_actual.get(period, {})
        pred_data = period_pred.get(period, {})

        if actual_data and pred_data:
            regions = set(actual_data.keys()) | set(pred_data.keys())
            diffs = {region: actual_data.get(region, 0) - pred_data.get(region, 0) for region in regions}
            period_diffs[period] = diffs
            all_diffs.extend(diffs.values())

    if not all_diffs:
        print("[WARN] No data available for choropleth plots")
        plt.close(fig)
        return

    # Create consistent color scale
    max_abs_diff = max(abs(min(all_diffs)), abs(max(all_diffs)))
    vmin, vmax = -max_abs_diff, max_abs_diff

    # Create mappable for colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Plot each period
    for i, period in enumerate(periods):
        ax = axes[i]
        diffs = period_diffs.get(period, {})

        if diffs:
            shp_copy = shp.copy()
            shp_copy["case_diff"] = shp_copy["region"].map(diffs)

            shp_copy.plot(column="case_diff", ax=ax, cmap="RdBu", vmin=vmin, vmax=vmax, legend=False)
            ax.set_title(period)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(period)
            ax.axis("off")

    # Add shared colorbar below the plots
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=30, location="bottom", pad=0.15)
    cbar.ax.text(-0.1, 0.5, "Obs < pred", ha="right", va="center", transform=cbar.ax.transAxes)
    cbar.ax.text(1.1, 0.5, "Obs > pred", ha="left", va="center", transform=cbar.ax.transAxes)

    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.95)

    # Adjust layout to make room for the colorbar
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_case_diff_choropleth(shp, actual_cases, pred_cases, output_path, title="Case Count Difference"):
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

    # Map the differences using region
    shp_copy["case_diff"] = shp_copy["region"].map(differences)

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
    Generate a shapefile from the model configuration with proper regional groupings.

    Args:
        model_config (dict): Model configuration dictionary containing region information

    Returns:
        tuple: (GeoDataFrame, dict) The processed shapefile with region-level geometries and region lookup dictionary
    """
    # Extract region information from config
    regions = model_config.get("regions", [])
    if not regions:
        raise ValueError("No regions specified in model config")
    admin_level = model_config.get("admin_level", None)
    summary_config = model_config.get("summary_config", {})

    # Get dot names for the regions
    dot_names = lp.find_matching_dot_names(
        regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv", admin_level=admin_level, verbose=0
    )

    # Load and filter shapefile
    shp = gpd.read_file(lp.root / "data/shp_africa_low_res.gpkg", layer="adm2")
    shp = shp[shp["dot_name"].isin(dot_names)]
    shp = shp.set_index("dot_name").loc[dot_names].reset_index()  # Ensure correct ordering

    if admin_level == 2:
        return shp

    elif admin_level == 1:
        shp["geometry"] = shp["geometry"].buffer(0)  # Fix topology issues
        shp = shp.dissolve(by="adm01", aggfunc="first").reset_index()  # Dissolve by adm01
        return shp

    elif admin_level == 0 and "region_groupings" not in summary_config:
        shp["geometry"] = shp["geometry"].buffer(0)  # Fix topology issues
        shp = lp.add_regional_groupings(shp)  # Add region column
        shp = shp.dissolve(by="region", aggfunc="first").reset_index()  # Dissolve by adm0
        shp = shp[["region", "geometry"]]
        return shp

    elif admin_level == 0 and "region_groupings" in summary_config:
        shp = lp.add_regional_groupings(shp, summary_config["region_groupings"])  # Apply regional groupings

        # Step 1: Dissolve by region to group polygons
        region_dissolved = shp.dissolve(by="region", as_index=False)

        # Step 2: For each region, fully merge all geometry parts into a single polygon
        def unify_region_geometry(region_df):
            return unary_union(region_df.geometry)

        unified_geoms = []
        for region_name in region_dissolved["region"]:
            region_geom = unary_union(shp[shp["region"] == region_name].geometry)
            unified_geoms.append((region_name, region_geom))

        # Step 3: Build new GeoDataFrame
        region_shp = gpd.GeoDataFrame(unified_geoms, columns=["region", "geometry"], crs=shp.crs)

        def extract_outer_shell(geom):
            # If it's a MultiPolygon, merge and take the union of all exteriors
            if geom.geom_type == "MultiPolygon":
                merged = unary_union(geom)
                largest = max(merged.geoms, key=lambda g: g.area)
                return Polygon(largest.exterior)
            elif geom.geom_type == "Polygon":
                return Polygon(geom.exterior)
            else:
                return geom  # Fallback (shouldn't happen)

        # Step 4: Extract outer shell of each region
        unified_geoms = []
        for region_name in region_dissolved["region"]:
            merged_geom = unary_union(shp[shp["region"] == region_name].geometry)
            outer_geom = extract_outer_shell(merged_geom)
            unified_geoms.append((region_name, outer_geom))
        region_shp = gpd.GeoDataFrame(unified_geoms, columns=["region", "geometry"], crs=shp.crs)

        # Step 5: Add label points
        region_shp_proj = region_shp.to_crs(epsg=3857)  # Reproject to projected CRS (meters)
        region_shp["center_lon"] = region_shp_proj.geometry.centroid.to_crs(epsg=4326).x
        region_shp["center_lat"] = region_shp_proj.geometry.centroid.to_crs(epsg=4326).y

        return region_shp


def get_trial_by_number(study, trial_number):
    """Get a trial by its number."""
    for trial in study.trials:
        if trial.number == trial_number:
            return trial
    raise ValueError(f"Trial {trial_number} not found in study")


def plot_trial_targets(study, trial_number, output_dir=None, shp=None, model_config=None, start_year=2018):
    """Plot targets for a specific trial number."""
    trial = get_trial_by_number(study, trial_number)
    if trial.state != optuna.trial.TrialState.COMPLETE:
        raise ValueError(f"Trial {trial_number} is not completed")

    actual = trial.user_attrs["actual"]
    preds = trial.user_attrs["predicted"]

    # Use provided model_config or default empty dict
    if model_config is None:
        model_config = {}

    # Use the output_dir directly instead of creating a subdirectory
    trial_dir = Path(output_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Generate shapefile if not provided
    if shp is None:
        try:
            shp = get_shapefile_from_config(model_config)
            print("[INFO] Generated shapefile from model config")
        except Exception as e:
            print(f"[WARN] Could not generate shapefile: {e}")
            shp = None

    # Use the same plotting logic as plot_targets but with trial-specific data
    _plot_targets_impl(actual, preds, trial_dir, shp, model_config, start_year, f"Trial {trial_number}")


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
    start_year = model_config.get("start_year", 2018)

    # Create output directory for best trial plots
    best_dir = Path(output_dir) / "best_trial_plots"
    best_dir.mkdir(exist_ok=True)

    # Generate shapefile if not provided
    if shp is None:
        try:
            shp = get_shapefile_from_config(model_config)
            print("[INFO] Generated shapefile from model config")
        except Exception as e:
            print(f"[WARN] Could not generate shapefile: {e}")
            shp = None

    # Use the common plotting implementation
    _plot_targets_impl(actual, preds, best_dir, shp, model_config, start_year, "Best")


def _plot_targets_impl(actual, preds, output_dir, shp, model_config, start_year, title_prefix):
    """
    Common implementation for plotting targets data.

    This function generates plots comparing actual and predicted target data for a given trial or set of trials.
    It is used by both `plot_targets` (for the best trial) and `plot_trial_targets` (for a specific trial).

    Parameters
    ----------
    actual : dict
        Dictionary containing the actual target data, typically with keys for different target types (e.g., "cases_by_period").
    preds : list or dict
        Predicted data, typically a list of dictionaries (one per replicate) or a dictionary with similar structure to `actual`.
    output_dir : Path or str
        Directory where the generated plots will be saved.
    shp : GeoDataFrame or None
        Shapefile data as a GeoPandas GeoDataFrame, or None if not available.
    model_config : dict
        Model configuration dictionary, may contain additional metadata such as "start_year".
    start_year : int
        The starting year for the time series plots.
    title_prefix : str
        Prefix to use in plot titles (e.g., "Best" or "Trial 5").

    Returns
    -------
    None
        The function saves plots to the specified output directory and does not return a value.
    """
    # For now, just call the original plot_targets logic with proper parameters
    # This is a simplified implementation - the full plotting logic from plot_targets
    # could be moved here for better code reuse
    print(f"[INFO] Plotting targets for {title_prefix} trial to {output_dir}")

    # Define consistent colors
    n_reps = len(preds)
    labels = ["Actual"] + [f"Rep {i + 1}" for i in range(n_reps)]
    cmap = cm.get_cmap("Dark2")
    color_map = {label: cmap(i) for i, label in enumerate(labels)}

    # For now, just implement the basic cases_by_period plot as an example
    if "cases_by_period" in actual:
        period_labels = list(actual["cases_by_period"].keys())
        x = np.arange(len(period_labels))
        actual_vals = [actual["cases_by_period"][period] for period in period_labels]

        plt.figure(figsize=(10, 6))
        plt.title(f"Cases by Period - {title_prefix}")
        plt.bar(x, actual_vals, width=0.6, edgecolor=color_map["Actual"], facecolor="none", linewidth=1.5, label="Actual")
        for i, rep in enumerate(preds):
            pred = rep["cases_by_period"]
            label = f"Rep {i + 1}"
            pred_vals = [pred.get(period, 0) for period in period_labels]
            plt.scatter(x, pred_vals, label=label, color=color_map[f"Rep {i + 1}"], marker="o", s=50)
        plt.xticks(x, period_labels, rotation=45, ha="right")
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_{title_prefix.lower()}_cases_by_period.png", bbox_inches="tight")
        plt.close()

    # Monthly Timeseries Cases
    if "cases_by_month" in actual:
        n_months = len(actual["cases_by_month"])
        months_series = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq="MS")
        plt.figure()
        plt.title(f"Cases by Month - {title_prefix}")
        plt.plot(months_series, actual["cases_by_month"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
        for i, rep in enumerate(preds):
            label = f"Rep {i + 1}"
            plt.plot(months_series, rep["cases_by_month"], "o-", label=label, color=color_map[label])
        plt.xlabel("Month")
        plt.ylabel("Cases")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_{title_prefix.lower()}_cases_by_month.png")
        plt.close()

    # Regional Cases (bar plot)
    if "cases_by_region" in actual:
        region_labels = list(actual["cases_by_region"].keys())
        x = np.arange(len(region_labels))
        width = 0.6
        plt.figure(figsize=(12, 8))
        plt.title(f"Regional Cases - {title_prefix}")
        plt.bar(
            x, actual["cases_by_region"].values(), width, label="Actual", edgecolor=color_map["Actual"], facecolor="none", linewidth=1.5
        )
        for i, rep in enumerate(preds):
            label = f"Rep {i + 1}"
            plt.scatter(x, rep["cases_by_region"].values(), label=f"Rep {i + 1}", color=color_map[label], marker="o", s=50)
        plt.xticks(x + width * (len(preds) // 2), region_labels, rotation=45, ha="right")
        plt.ylabel("Cases")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_{title_prefix.lower()}_cases_by_region.png")
        plt.close()

    # Regional Cases by Period (nested dictionary structure)
    if "cases_by_region_period" in actual:
        region_period_data = actual["cases_by_region_period"]

        # Extract all unique periods from all regions
        all_periods = set()
        for region_dict in region_period_data.values():
            all_periods.update(region_dict.keys())
        periods = sorted(all_periods)

        # Extract all unique regions
        regions = sorted(region_period_data.keys())

        # Create figure with subplots stacked vertically (one per period)
        n_periods = len(periods)
        if n_periods > 0:
            fig, axes = plt.subplots(n_periods, 1, figsize=(12, 4 * n_periods))
            if n_periods == 1:
                axes = [axes]

            for i, period in enumerate(periods):
                ax = axes[i]

                # Extract data for this period across all regions
                x = np.arange(len(regions))
                actual_vals = [region_period_data.get(region, {}).get(period, 0) for region in regions]

                # Plot actual as outlined bar
                ax.bar(x, actual_vals, width=0.6, edgecolor=color_map["Actual"], facecolor="none", linewidth=1.5, label="Actual")

                # Plot predicted reps as colored dots
                for j, rep in enumerate(preds):
                    label = f"Rep {j + 1}"
                    rep_data = rep.get("cases_by_region_period", {})
                    rep_vals = [rep_data.get(region, {}).get(period, 0) for region in regions]
                    ax.scatter(x, rep_vals, label=label, color=color_map[label], marker="o", s=50)

                ax.set_title(f"Regional Cases - {period}")
                ax.set_xticks(x)
                ax.set_xticklabels(regions, rotation=45, ha="right")
                ax.set_ylabel("Cases")
                ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / f"plot_{title_prefix.lower()}_cases_by_region_period.png")
            plt.close()

    if "cases_by_region_month" in actual:
        cases_by_region_month_actual = actual.get("cases_by_region_month")
        regions = list(cases_by_region_month_actual.keys())
        n_regions = len(regions)

        if n_regions > 0:
            # Create subplot grid (2x2 for 4 regions, adjust if different number)
            n_cols = 2
            n_rows = (n_regions + n_cols - 1) // n_cols  # Ceiling division

            # Define dynamic figure size: scale height per row and keep width fixed
            fig_height_per_row = 3.5
            fig_width = 15
            fig_height = n_rows * fig_height_per_row
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            fig.suptitle("Regional Monthly Timeseries Comparison", fontsize=16)

            # Flatten axes for easier indexing if multiple rows
            # if n_regions > 1:
            #     axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            # else:
            #     axes = [axes]

            # Normalize axes to always be a flat list of Axes
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

            # Get the length of the first key in the dictionary
            first_key = next(iter(actual["cases_by_region_month"]))
            n_months = len(actual["cases_by_region_month"][first_key])
            months_series = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq="MS")

            for idx, region in enumerate(regions):
                ax = axes[idx]
                timeseries = cases_by_region_month_actual[region]

                # Plot actual data
                ax.plot(months_series, timeseries, "o-", label="Actual", color=color_map["Actual"], linewidth=2)

                # Add predicted data for each replicate
                for i, rep in enumerate(preds):
                    if "cases_by_region_month" in rep and region in rep["cases_by_region_month"]:
                        label = f"Rep {i + 1}"
                        rep_timeseries = rep["cases_by_region_month"][region]
                        ax.plot(months_series, rep_timeseries, "o-", label=label, color=color_map[label])

                ax.set_title(f"{region.replace('_', ' ').title()}")
                ax.set_xlabel("Month")
                ax.set_ylabel("Cases")
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3)

                # Only add legend to first subplot to avoid clutter
                if idx == 0:
                    ax.legend(loc="upper left")

            # Hide any unused subplots
            for idx in range(n_regions, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(output_dir / f"plot_{title_prefix.lower()}_cases_by_region_month.png", dpi=300, bbox_inches="tight")
            plt.close()

    # District Case Bin Counts (binned histogram comparison)
    if "case_bins_by_region" in actual:
        # Read bin configuration from model config
        bin_config = model_config.get("summary_config", {}).get("case_bins", {})
        bin_labels = bin_config.get("bin_labels", ["0", "1", "2", "3", "4", "5-9", "10-19", "20+"])

        case_bins_by_region_actual = actual["case_bins_by_region"]
        regions = list(case_bins_by_region_actual.keys())
        n_regions = len(regions)

        if n_regions > 0:
            # Create subplot grid (2x2 for 4 regions, adjust if different number)
            n_cols = 2
            n_rows = (n_regions + n_cols - 1) // n_cols  # Ceiling division

            fig_height_per_row = 3
            fig_width = 15
            fig_height = n_rows * fig_height_per_row
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            # fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
            fig.suptitle("District Case Count Distribution by Region", fontsize=16)

            # Flatten axes for easier indexing if multiple rows
            if n_regions > 1:
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            else:
                axes = [axes]

            # Get consistent y-axis scale across all subplots
            all_counts = []
            all_counts.extend(case_bins_by_region_actual.values())
            for rep in preds:
                if "case_bins_by_region" in rep:
                    all_counts.extend(rep["case_bins_by_region"].values())
            max_count = max(max(counts) if counts else 0 for counts in all_counts)

            for idx, region in enumerate(regions):
                ax = axes[idx]
                actual_counts = case_bins_by_region_actual[region]
                x_positions = range(len(bin_labels))

                # Plot actual data as bars with edges only
                ax.bar(x_positions, actual_counts, width=0.8, edgecolor=color_map["Actual"], facecolor="none", linewidth=2, label="Actual")

                # Add predicted data for each replicate as dots
                for i, rep in enumerate(preds):
                    if "case_bins_by_region" in rep and region in rep["case_bins_by_region"]:
                        label = f"Rep {i + 1}"
                        rep_counts = rep["case_bins_by_region"][region]
                        ax.scatter(x_positions, rep_counts, label=label, color=color_map[label], marker="o", s=50)

                ax.set_title(f"{region.replace('_', ' ').title()}")
                ax.set_xlabel("Number of Cases")
                ax.set_ylabel("Number of Districts")
                ax.set_xticks(x_positions)
                ax.set_xticklabels(bin_labels, rotation=0)
                ax.set_ylim(0, max_count * 1.1)
                ax.grid(True, alpha=0.3)

                # Add count annotations on actual bars
                for i, count in enumerate(actual_counts):
                    if count > 0:
                        ax.text(i, count + max_count * 0.02, f"{int(count)}", ha="center", va="bottom", fontsize=9)

                # Only add legend to first subplot to avoid clutter
                if idx == 0:
                    ax.legend(loc="upper right")

            # Hide any unused subplots
            for idx in range(n_regions, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(output_dir / f"plot_{title_prefix.lower()}_case_bins_by_region.png", dpi=300, bbox_inches="tight")
            plt.close()

    # Plot choropleth of case count differences for each replicate
    if shp is not None and "cases_by_region" in actual:
        for i, rep in enumerate(preds):
            if "cases_by_region" in rep:
                plot_case_diff_choropleth(
                    shp=shp,
                    actual_cases=actual["cases_by_region"],
                    pred_cases=rep["cases_by_region"],
                    output_path=output_dir / f"plot_{title_prefix.lower()}_case_diff_choropleth_rep{i + 1}.png",
                    title=f"Case Count Difference (Actual - Predicted) - Rep {i + 1}",
                )

    # Plot temporal choropleth of case count differences for each replicate
    if shp is not None and "cases_by_region_period" in actual:
        for i, rep in enumerate(preds):
            if "cases_by_region_period" in rep:
                plot_case_diff_choropleth_temporal(
                    shp=shp,
                    actual_cases_by_period=actual["cases_by_region_period"],
                    pred_cases_by_period=rep["cases_by_region_period"],
                    output_path=output_dir / f"plot_{title_prefix.lower()}_case_diff_choropleth_temporal_rep{i + 1}.png",
                    title=f"Case Count Difference by Period (Actual - Predicted) - Rep {i + 1}",
                )

    # Total Nodes with Cases
    if "nodes_with_cases_total" in actual:
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
        plt.savefig(output_dir / f"plot_{title_prefix.lower()}_nodes_with_cases_total.png")
        plt.close()

    # Monthly Nodes with Cases
    if "nodes_with_cases_timeseries" in actual:
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
        plt.savefig(output_dir / f"plot_{title_prefix.lower()}_nodes_with_cases_timeseries.png")
        plt.close()

    # Regional Cases (bar plot)
    if "regional" in actual:
        x = np.arange(len(region_labels))
        width = 0.1
        plt.figure()
        plt.title("Regional")
        plt.bar(x, [actual["regional"][r] for r in region_labels], width, label="Actual", color=color_map["Actual"])
        for i, rep in enumerate(preds):
            label = f"Rep {i + 1}"
            plt.bar(x + (i + 1) * width, [rep["regional"][r] for r in region_labels], width, label=f"Rep {i + 1}", color=color_map[label])
        plt.xticks(x + width * (len(preds) // 2), region_labels)
        plt.ylabel("Cases")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_{title_prefix.lower()}_regional.png")
        plt.close()

    if "regional_by_period" in actual:
        regional_by_period_actual = actual.get("regional_by_period")

        # Extract periods and regions from the dictionary keys
        periods = []
        period_data = {}

        for key, value in regional_by_period_actual.items():
            # Parse the tuple key format: "('NIGERIA:JIGAWA', '2018-2019')"
            # Extract both the region and period from the tuple-like string
            parts = key.strip("()").split("', '")
            if len(parts) == 2:
                adm01_name = parts[0].strip("'")
                period = parts[1].strip("'")

                if period not in periods:
                    periods.append(period)
                    period_data[period] = {}

                period_data[period][adm01_name] = value

        # Use the periods in the order they appear in the dictionary
        # Create figure with subplots stacked vertically
        n_periods = len(periods)
        if n_periods > 0:
            fig, axes = plt.subplots(n_periods, 1, figsize=(12, 4 * n_periods))
            if n_periods == 1:
                axes = [axes]

            for i, period in enumerate(periods):
                ax = axes[i]
                data = period_data.get(period, {})

                if data:
                    adm_labels = sorted(data.keys())
                    x = np.arange(len(adm_labels))
                    actual_vals = [data.get(adm, 0) for adm in adm_labels]

                    # Plot actual as outlined bar
                    ax.bar(x, actual_vals, width=0.6, edgecolor=color_map["Actual"], facecolor="none", linewidth=1.5, label="Actual")

                    # Plot predicted reps as colored dots
                    for j, rep in enumerate(preds):
                        label = f"Rep {j + 1}"
                        rep_data = {}
                        for key, value in rep.get("regional_by_period", {}).items():
                            # Parse the same way to extract period
                            parts = key.strip("()").split("', '")
                            if len(parts) == 2:
                                rep_adm01_name = parts[0].strip("'")
                                rep_period = parts[1].strip("'")
                                if rep_period == period:
                                    rep_data[rep_adm01_name] = value

                        rep_vals = [rep_data.get(adm, 0) for adm in adm_labels]
                        ax.scatter(x, rep_vals, label=label, color=color_map[label], marker="o", s=50)

                    ax.set_title(f"Regional Cases - {period}")
                    ax.set_xticks(x)
                    ax.set_xticklabels(adm_labels, rotation=45, ha="right")
                    ax.set_ylabel("Cases")
                    ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / f"plot_{title_prefix.lower()}_regional_by_period.png")
            plt.close()


def plot_likelihoods(study, output_dir=None, use_log=True, trial_number=None):
    # Default output directory to current working dir if not provided
    if output_dir:
        # If trial_number is specified, use output_dir directly, otherwise use optuna_plots subdirectory
        if trial_number is not None:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(output_dir) / "optuna_plots"
    else:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    if trial_number is not None:
        trial = get_trial_by_number(study, trial_number)
        likelihoods = trial.user_attrs["likelihoods"]
        title_suffix = f" - Trial {trial_number}"
    else:
        best = study.best_trial
        likelihoods = best.user_attrs["likelihoods"]
        title_suffix = " - Best Trial"
    exclude_keys = {"total_log_likelihood"}
    keys = [k for k in likelihoods if k not in exclude_keys]
    values = [likelihoods[k] for k in keys]

    fig, ax = plt.subplots(figsize=(12, 7))  # Increased height to accommodate labels
    bars = ax.bar(keys, values)  # noqa: F841
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel("Log Likelihood")
    else:
        ax.set_ylabel("Likelihood")
    ax.set_title(f"Calibration Log-Likelihoods by Component{title_suffix}")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    # Add text labels on bars after scale is set
    try:
        for bar in ax.patches:
            height = bar.get_height()
            if height > 0:  # Only add labels for positive values
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * (1.05 if not use_log else 0.95),  # Offset slightly from bar
                    f"{height:.1f}",
                    ha="center",
                    va="bottom" if not use_log else "top",
                    fontsize=9,
                )
    except Exception as e:
        print(f"[WARN] Could not add bar labels: {e}")
    plt.subplots_adjust(bottom=0.2)  # Reserve 20% of figure height for x-labels
    plt.savefig(output_dir / "plot_likelihoods.png", bbox_inches="tight")
    plt.close()
    plt.show()


def plot_runtimes(study, output_dir=None):
    # Default output directory to current working dir if not provided
    output_dir = Path(output_dir) / "optuna_plots" if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        plt.close()
        # plt.show()
    else:
        print("No completed trials with valid timestamps to plot.")


def load_region_group_labels(model_config_path):
    with open(model_config_path) as f:
        config = yaml.safe_load(f)
    region_groups = config.get("summary_config", {}).get("region_groups", {})
    return list(region_groups.keys())


def plot_multiple_choropleths(shp, node_lookup, actual_cases, trial_predictions, output_path, n_cols=5, legend_position="bottom"):
    """
    Plot multiple choropleths in a grid layout showing differences between actual and predicted cases.

    Args:
        shp (GeoDataFrame): The shapefile GeoDataFrame
        node_lookup (dict): Dictionary mapping dot_names to administrative regions
        actual_cases (dict): Dictionary of actual case counts by region
        trial_predictions (list): List of (trial_number, value, predictions) tuples
        output_path (Path): Path to save the plot
        n_cols (int): Number of columns in the grid
        legend_position (str): Position of the legend ("bottom" or "right")
    """
    n_trials = len(trial_predictions)
    n_rows = (n_trials + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with extra space at the bottom for the colorbar
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 1))

    # Calculate global min/max for consistent color scale
    all_diffs = []
    for _, _, pred_cases in trial_predictions:
        diffs = [actual_cases.get(region, 0) - pred_cases.get(region, 0) for region in set(actual_cases.keys()) | set(pred_cases.keys())]
        all_diffs.extend(diffs)

    max_abs_diff = max(abs(min(all_diffs)), abs(max(all_diffs)))
    vmin, vmax = -max_abs_diff, max_abs_diff

    # Create subplot grid that leaves space for the colorbar
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[*[1] * n_rows, 0.1])

    for idx, (trial_number, value, pred_cases) in enumerate(trial_predictions):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Calculate differences for this trial
        shp_copy = shp.copy()
        differences = {
            region: actual_cases.get(region, 0) - pred_cases.get(region, 0) for region in set(actual_cases.keys()) | set(pred_cases.keys())
        }
        shp_copy["case_diff"] = shp_copy["adm01_name"].map(differences)

        # Plot the map
        shp_copy.plot(column="case_diff", ax=ax, cmap="RdBu", vmin=vmin, vmax=vmax, legend=False)

        ax.set_title(f"Trial {trial_number}\nValue: {value:.2f}")
        ax.axis("off")

    # Add a single colorbar at the bottom
    cbar_ax = fig.add_subplot(gs[-1, :])
    cbar_ax.axis("off")

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap="RdBu")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    # Add annotations to the left and right of the colorbar
    cbar.ax.text(-0.1, 0.5, "Obs < pred", ha="right", va="center", transform=cbar.ax.transAxes)
    cbar.ax.text(1.1, 0.5, "Obs > pred", ha="left", va="center", transform=cbar.ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_top_trials(study, output_dir, n_best=10, title="Top Calibration Results", shp=None, node_lookup=None, start_year=2018):
    """
    Plot the top n best calibration trials using the same visualizations as plot_targets.

    Args:
        study (optuna.Study): The Optuna study containing trials
        output_dir (Path): Directory to save plots
        n_best (int): Number of best trials to plot
        title (str): Title for the plot
        shp (GeoDataFrame, optional): Shapefile for choropleth plots
        node_lookup (dict, optional): Dictionary mapping dot_names to administrative regions
    """
    # Get trials sorted by value (ascending)
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    top_trials = trials[:n_best]

    # Load metadata and model config
    metadata_path = Path(output_dir) / "study_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"study_metadata.json not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_config = metadata.get("model_config", {})

    # Generate shapefile if not provided
    if shp is None:
        try:
            shp, node_lookup = get_shapefile_from_config(model_config)
            print("[INFO] Generated shapefile from model config")
        except Exception as e:
            print(f"[WARN] Could not generate shapefile: {e}")
            shp = None

    # Define consistent colors for trials
    cmap = cm.get_cmap("tab20")
    color_map = {f"Trial {trial.number}": cmap(i) for i, trial in enumerate(top_trials)}

    # Create output directory for top trials plots
    top_trials_dir = Path(output_dir) / "top_10_trial_plots"
    top_trials_dir.mkdir(exist_ok=True)

    # Get actual data from first trial (should be same for all)
    actual = top_trials[0].user_attrs["actual"]

    # Total Infected
    if "total_infected" in actual:
        plt.figure()
        plt.title(f"Total Infected - Top {n_best} Trials")
        width = 0.8 / (n_best + 1)  # Adjust bar width based on number of trials
        x = np.arange(2)  # Just two bars: Actual and Predicted
        plt.bar(x[0], actual["total_infected"][0], width, label="Actual", color="black")
        for i, trial in enumerate(top_trials):
            pred = trial.user_attrs["predicted"][0]  # Get first replicate
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            plt.bar(
                x[1] + (i - n_best / 2) * width, pred["total_infected"][0], width, label=label, color=color_map[f"Trial {trial.number}"]
            )
        plt.xticks(x, ["Actual", "Predicted"])
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "total_infected_comparison.png", bbox_inches="tight")
        plt.close()

    # Yearly Cases
    if "yearly_cases" in actual:
        years = list(range(start_year, start_year + len(actual["yearly_cases"])))
        plt.figure(figsize=(10, 6))
        plt.title(f"Yearly Cases - Top {n_best} Trials")
        plt.plot(years, actual["yearly_cases"], "o-", label="Actual", color="black", linewidth=2)
        for trial in top_trials:
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            plt.plot(years, pred["yearly_cases"], "o--", label=label, color=color_map[f"Trial {trial.number}"])
        plt.xlabel("Year")
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "yearly_cases_comparison.png", bbox_inches="tight")
        plt.close()

    # Monthly Cases
    if "monthly_cases" in actual:
        months = list(range(1, 1 + len(actual["monthly_cases"])))
        plt.figure(figsize=(10, 6))
        plt.title(f"Monthly Cases - Top {n_best} Trials")
        plt.plot(months, actual["monthly_cases"], "o-", label="Actual", color="black", linewidth=2)
        for trial in top_trials:
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            plt.plot(months, pred["monthly_cases"], "o--", label=label, color=color_map[f"Trial {trial.number}"])
        plt.xlabel("Month")
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "monthly_cases_comparison.png", bbox_inches="tight")
        plt.close()

    # Monthly Timeseries
    if "monthly_timeseries" in actual:
        n_months = len(actual["monthly_timeseries"])
        months_series = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq="MS")
        plt.figure(figsize=(10, 6))
        plt.title(f"Monthly Timeseries - Top {n_best} Trials")
        plt.plot(months_series, actual["monthly_timeseries"], "o-", label="Actual", color="black", linewidth=2)
        for trial in top_trials:
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            plt.plot(months_series, pred["monthly_timeseries"], "o--", label=label, color=color_map[f"Trial {trial.number}"])
        plt.xlabel("Month")
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "monthly_timeseries_comparison.png", bbox_inches="tight")
        plt.close()

    # Total by Period if available
    total_by_period_actual = actual.get("total_by_period")
    if total_by_period_actual:
        # Use the keys from the dictionary in their natural order
        period_labels = list(actual["total_by_period"].keys())
        x = np.arange(len(period_labels))
        actual_vals = [actual["total_by_period"][period] for period in period_labels]

        plt.figure(figsize=(10, 6))
        plt.title(f"Total Cases by Period - Top {n_best} Trials")
        plt.bar(x, actual_vals, width=0.6, edgecolor="black", facecolor="none", linewidth=1.5, label="Actual")

        for trial in top_trials:
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            pred_vals = [pred["total_by_period"].get(period, 0) for period in period_labels]
            plt.scatter(x, pred_vals, label=label, color=color_map[f"Trial {trial.number}"], marker="o", s=50)

        plt.xticks(x, period_labels, rotation=45, ha="right")
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "total_by_period_comparison.png", bbox_inches="tight")
        plt.close()

    # ADM0 Cases if available
    adm0_actual = actual.get("adm0_cases")
    if adm0_actual:
        adm_labels = sorted(actual["adm0_cases"].keys())
        x = np.arange(len(adm_labels))
        actual_vals = [actual["adm0_cases"].get(adm, 0) for adm in adm_labels]

        plt.figure(figsize=(12, 6))
        plt.title(f"ADM0 Cases - Top {n_best} Trials")
        plt.bar(x, actual_vals, width=0.6, edgecolor="gray", facecolor="none", linewidth=1.5, label="Actual")

        for trial in top_trials:
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            pred_vals = [pred["adm0_cases"].get(adm, 0) for adm in adm_labels]
            plt.scatter(x, pred_vals, label=label, color=color_map[f"Trial {trial.number}"], marker="o", s=50)

        plt.xticks(x, adm_labels, rotation=45, ha="right")
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "adm0_cases_comparison.png", bbox_inches="tight")
        plt.close()

    # ADM01 Cases if available
    adm01_actual = actual.get("adm01_cases")
    if adm01_actual:
        adm_labels = sorted(actual["adm01_cases"].keys())
        x = np.arange(len(adm_labels))
        actual_vals = [actual["adm01_cases"].get(adm, 0) for adm in adm_labels]

        plt.figure(figsize=(12, 6))
        plt.title(f"ADM01 Regional Cases - Top {n_best} Trials")
        plt.bar(x, actual_vals, width=0.6, edgecolor="black", facecolor="none", linewidth=1.5, label="Actual")

        for trial in top_trials:
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            pred_vals = [pred["adm01_cases"].get(adm, 0) for adm in adm_labels]
            plt.scatter(x, pred_vals, label=label, color=color_map[f"Trial {trial.number}"], marker="o", s=50)

        plt.xticks(x, adm_labels, rotation=45, ha="right")
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "adm01_cases_comparison.png", bbox_inches="tight")
        plt.close()

    # Regional Cases
    if "regional_cases" in actual:
        region_labels = list(model_config.get("summary_config", {}).get("region_groups", {}).keys())
        x = np.arange(len(region_labels))
        width = 0.8 / (n_best + 1)

        plt.figure(figsize=(12, 6))
        plt.title(f"Regional Cases - Top {n_best} Trials")
        plt.bar(x, actual["regional_cases"], width, label="Actual", color="black")

        for i, trial in enumerate(top_trials):
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            plt.bar(x + (i + 1) * width, pred["regional_cases"], width, label=label, color=color_map[f"Trial {trial.number}"])

        plt.xticks(x + width * (n_best / 2), region_labels)
        plt.ylabel("Cases")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "regional_cases_comparison.png", bbox_inches="tight")
        plt.close()

    # Total Nodes with Cases
    if "nodes_with_cases_total" in actual:
        plt.figure()
        plt.title(f"Total Nodes with Cases - Top {n_best} Trials")
        width = 0.8 / (n_best + 1)
        x = np.arange(2)  # Just two categories: Actual and Predicted
        plt.bar(x[0], actual["nodes_with_cases_total"][0], width, label="Actual", color="black")
        for i, trial in enumerate(top_trials):
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            plt.bar(
                x[1] + (i - n_best / 2) * width,
                pred["nodes_with_cases_total"][0],
                width,
                label=label,
                color=color_map[f"Trial {trial.number}"],
            )
        plt.xticks(x, ["Actual", "Predicted"])
        plt.ylabel("Nodes")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "nodes_with_cases_total_comparison.png", bbox_inches="tight")
        plt.close()

    # Monthly Nodes with Cases
    if "nodes_with_cases_timeseries" in actual:
        n_months = len(actual["nodes_with_cases_timeseries"])
        months = list(range(1, n_months + 1))
        plt.figure(figsize=(10, 6))
        plt.title(f"Monthly Nodes with Cases - Top {n_best} Trials")
        plt.plot(months, actual["nodes_with_cases_timeseries"], "o-", label="Actual", color="black", linewidth=2)
        for trial in top_trials:
            pred = trial.user_attrs["predicted"][0]
            label = f"Trial {trial.number} (value={trial.value:.2f})"
            plt.plot(months, pred["nodes_with_cases_timeseries"], "o-", label=label, color=color_map[f"Trial {trial.number}"])
        plt.xlabel("Month")
        plt.ylabel("Number of Nodes with ≥1 Case")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(top_trials_dir / "nodes_with_cases_timeseries_comparison.png", bbox_inches="tight")
        plt.close()

    # Plot choropleth of case count differences for all trials in one figure
    if shp is not None and node_lookup is not None:
        actual = top_trials[0].user_attrs["actual"]
        if "adm01_cases" in actual:
            trial_predictions = [(trial.number, trial.value, trial.user_attrs["predicted"][0]["adm01_cases"]) for trial in top_trials]
            plot_multiple_choropleths(
                shp=shp,
                node_lookup=node_lookup,
                actual_cases=actual["adm01_cases"],
                trial_predictions=trial_predictions,
                output_path=top_trials_dir / "case_diff_choropleths.png",
                legend_position="bottom",  # Add parameter to control legend position
            )

        # Plot temporal choropleth for the best trial
        if "cases_by_region_period" in actual:
            best_trial = top_trials[0]
            best_pred = best_trial.user_attrs["predicted"][0]
            plot_case_diff_choropleth_temporal(
                shp=shp,
                actual_cases_by_period=actual["cases_by_region_period"],
                pred_cases_by_period=best_pred["cases_by_region_period"],
                output_path=top_trials_dir / "case_diff_choropleth_temporal_best.png",
                title=f"Case Count Difference by Period - Best Trial {best_trial.number} (value={best_trial.value:.2f})",
            )
