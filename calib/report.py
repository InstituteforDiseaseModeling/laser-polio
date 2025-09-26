import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine

# Ensure we're using the Agg backend for better cross-platform compatibility
matplotlib.use("Agg")

import optuna
import optuna.visualization as vis
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


def get_top_trials(study, n=10, include_params=True, include_user_attrs=True):
    """
    Efficiently retrieve the top N trials from an Optuna MySQL database.

    Parameters
    ----------
    study : optuna.Study
        The Optuna study object (should have storage_url and study_name attributes)
    n : int, default=10
        Number of top trials to retrieve
    include_params : bool, default=True
        Whether to include trial parameters
    include_user_attrs : bool, default=True
        Whether to include user attributes

    Returns
    -------
    pd.DataFrame
        DataFrame containing trial information, parameters, and user attributes
    """

    # Use the storage_url you attached to the study object
    if hasattr(study, "storage_url"):
        storage_url = study.storage_url
    else:
        raise ValueError("Study object missing storage_url attribute. Please set: study.storage_url = cfg.storage_url")

    engine = create_engine(storage_url)

    try:
        study_id = study._study_id

        # Get top trials with values
        query = """
            SELECT t.trial_id, t.number, tv.value, t.datetime_start, t.datetime_complete
            FROM trials t
            INNER JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = %(study_id)s 
                AND t.state = 'COMPLETE'
                AND tv.objective = 0
            ORDER BY tv.value ASC
            LIMIT %(n)s
        """

        df = pd.read_sql(query, engine, params={"study_id": study_id, "n": n})

        if len(df) == 0:
            print(f"Warning: No completed trials found in study '{study.study_name}'")
            return df

        trial_ids = df["trial_id"].tolist()
        trial_ids_str = ",".join(map(str, trial_ids))

        # Get parameters if requested
        if include_params:
            # Validate that all trial_ids are integers (they should be from the DB)
            validated_ids = [int(tid) for tid in trial_ids]
            trial_ids_str = ",".join(map(str, validated_ids))

            params_query = f"""
                SELECT trial_id, param_name, param_value
                FROM trial_params
                WHERE trial_id IN ({trial_ids_str})
            """  # noqa: S608 - trial_ids are validated integers from database
            df_params = pd.read_sql(params_query, engine)

            if not df_params.empty:
                df_params_wide = df_params.pivot(index="trial_id", columns="param_name", values="param_value")
                df = df.merge(df_params_wide, left_on="trial_id", right_index=True, how="left")

        # Get user attributes if requested
        if include_user_attrs:
            user_attrs_query = f"""
                SELECT trial_id, `key`, value_json
                FROM trial_user_attributes
                WHERE trial_id IN ({trial_ids_str})
            """  # noqa: S608 - trial_ids are validated integers from database
            df_user_attrs = pd.read_sql(user_attrs_query, engine)

            if not df_user_attrs.empty:
                # Parse JSON values
                df_user_attrs["value_parsed"] = df_user_attrs["value_json"].apply(json.loads)

                # Pivot to wide format
                df_attrs_wide = df_user_attrs.pivot(index="trial_id", columns="key", values="value_parsed")
                df = df.merge(df_attrs_wide, left_on="trial_id", right_index=True, how="left")

        # Sort by value to maintain order
        df = df.sort_values("value").reset_index(drop=True)

        return df

    finally:
        engine.dispose()


def plot_targets(study, n=1, output_dir=None, shp=None):
    # Load top trials
    trials = get_top_trials(study, n=n)
    actual = trials.iloc[0]["actual"]  # Extract actual (should be same for all trials)
    preds = trials["predicted"].tolist()  # List of predicted arrays for each trial

    # Load metadata and model config
    metadata_path = Path(output_dir) / "study_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"study_metadata.json not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_config = metadata.get("model_config", {})
    start_year = model_config["start_year"]

    # Generate shapefile if not provided
    if shp is None:
        try:
            shp = get_shapefile_from_config(model_config)
            print("[INFO] Generated shapefile from model config")
        except Exception as e:
            print(f"[WARN] Could not generate shapefile: {e}")
            shp = None

    # Define consistent colors using seaborn's 'flare' palette
    # Option 1: Use seaborn color palette directly
    colors = sns.color_palette("Reds", n_colors=len(trials))
    colors.reverse()  # Reverse so darkest is for best trials
    # Create color map with Trial objects for consistent use across all plots
    color_map = {}
    for i, (_idx, row) in enumerate(trials.iterrows()):
        trial_num = row["number"]
        color_map[f"Trial {trial_num}"] = colors[i]
        # Also store by index for convenience
        color_map[i] = colors[i]
    # Add standard colors for actual/predicted labels
    color_map["Actual"] = "black"
    color_map["Predicted"] = colors[0]  # Darkest flare color for best/single prediction

    # Create output directory
    output_path = Path(output_dir) / "target_plots"
    output_path.mkdir(exist_ok=True, parents=True)

    # Plotting
    plot_cases_total(actual, preds, trials, output_path, color_map)
    plot_cases_by_period(actual, preds, trials, output_path, color_map)
    plot_cases_by_month(actual, preds, trials, output_path, color_map)
    plot_cases_by_month_timeseries(actual, preds, trials, output_path, color_map, start_year)
    plot_cases_by_region(actual, preds, trials, output_path, color_map)
    plot_cases_by_region_period(actual, preds, trials, output_path, color_map)
    plot_cases_by_region_month(actual, preds, trials, output_path, color_map, start_year)
    plot_case_bins_by_region(actual, preds, trials, output_path, color_map, model_config)
    plot_case_diff_choropleth_multi(actual, preds, trials, output_path, color_map, shp, model_config)


def plot_cases_total(actual, preds, trials, output_dir, color_map):
    if "cases_total" not in actual:
        print("[WARN] cases_total not found in calib targets. Skipping plot.")
        return

    # Extract predicted values
    predicted_values = [pred[0]["cases_total"] for pred in preds]
    actual_value = actual["cases_total"]

    if len(preds) == 1:
        # --- Two-bar comparison (Actual vs Predicted) ---
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("Cases Total", fontsize=14, fontweight="bold")

        x = np.arange(2)
        labels = ["Actual", "Predicted"]
        values = [actual_value, predicted_values[0]]
        colors = [color_map["Actual"], color_map["Predicted"]]

        bars = ax.bar(x, values, width=0.5, color=colors, edgecolor="darkgrey", linewidth=1.2, alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, values, strict=False):
            if np.isfinite(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.0f}", ha="center", va="bottom", fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Cases", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_dir / "cases_total.png", bbox_inches="tight", dpi=150)
        plt.close()

    else:
        # --- Histogram for multiple trials ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Cases Total Distribution", fontsize=14, fontweight="bold")

        # Calculate better bin size based on data range
        n_bins = min(30, max(10, int(np.sqrt(len(predicted_values)) * 2)))

        # Plot histogram with consistent grey coloring
        ax.hist(predicted_values, bins=n_bins, color="lightgrey", edgecolor="darkgrey", alpha=0.6, linewidth=0.8)

        # Add vertical lines
        ax.axvline(actual_value, color="black", linestyle="-", linewidth=2.5, label=f"Actual: {actual_value:.0f}", zorder=5)

        best_pred = predicted_values[0]
        best_trial_num = trials.iloc[0]["number"]
        ax.axvline(
            best_pred, color=color_map[0], linestyle="--", linewidth=2.5, label=f"Best (Trial {best_trial_num}): {best_pred:.0f}", zorder=4
        )

        # Formatting
        ax.set_xlabel("Cases Total", fontsize=12)
        ax.set_ylabel("Number of Trials", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        # Compact legend
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "cases_total.png", bbox_inches="tight", dpi=150)
        plt.close()


def plot_cases_by_period(actual, preds, trials, output_dir, color_map):
    if "cases_by_period" not in actual:
        print("[WARN] cases_by_period not found in calib targets. Skipping plot.")
        return

    # Extract predicted values
    predicted_values = [pred[0]["cases_by_period"] for pred in preds]
    actual_value = actual["cases_by_period"]

    period_labels = list(actual_value.keys())
    x = np.arange(len(period_labels))
    actual_vals = [actual_value[period] for period in period_labels]

    fig, ax = plt.subplots(figsize=(10, 6))  # Slightly smaller width
    ax.set_title("Cases by Period", fontsize=14, fontweight="bold")

    # Plot actual data as grey bars
    _bars = ax.bar(x, actual_vals, width=0.7, color="lightgrey", alpha=0.6, edgecolor="lightgrey", linewidth=1.2, label="Actual", zorder=1)

    if len(preds) == 1:
        # Single prediction
        trial_num = trials.iloc[0]["number"]
        pred_vals = [predicted_values[0].get(period, 0) for period in period_labels]
        ax.plot(x, pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Trial {trial_num}", zorder=3)
    else:
        # Multiple predictions - simplified approach
        # Plot all non-best trials in one go for efficiency
        for i in range(1, len(preds)):  # Skip best (index 0)
            pred_vals = [predicted_values[i].get(period, 0) for period in period_labels]
            # Graduated transparency based on ranking
            alpha = 0.2 + (0.3 * (1 - i / len(preds)))
            ax.plot(x, pred_vals, "-", color=color_map[i], linewidth=1.0, alpha=alpha, zorder=2)

        # Add single legend entry for other trials
        ax.plot([], [], "-", color="grey", alpha=0.4, linewidth=1.0, label=f"Trials 2-{len(preds)} (n={len(preds) - 1})")

        # Plot best trial last with markers
        best_trial_num = trials.iloc[0]["number"]
        best_pred_vals = [predicted_values[0].get(period, 0) for period in period_labels]
        ax.plot(x, best_pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Best: Trial {best_trial_num}", zorder=4)

    # Formatting improvements
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Cases", fontsize=12)
    ax.set_xlabel("Period", fontsize=12)

    # Cleaner grid
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax.set_axisbelow(True)  # Grid behind data

    # Tighter legend
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Add subtle formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "cases_by_period.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_cases_by_month(actual, preds, trials, output_dir, color_map):
    if "cases_by_month" not in actual:
        print("[WARN] cases_by_month not found in calib targets. Skipping plot.")
        return

    # Extract predicted values
    predicted_values = [pred[0]["cases_by_month"] for pred in preds]
    actual_values = actual["cases_by_month"]

    # Create month labels (1-indexed)
    months = list(range(1, 1 + len(actual_values)))
    x = np.array(months) - 1  # Convert to 0-indexed for plotting positions

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cases by Month (Aggregated Across Years)", fontsize=14, fontweight="bold")

    # Plot actual data as grey bars
    _bars = ax.bar(
        x, actual_values, width=0.7, color="lightgrey", alpha=0.6, edgecolor="lightgrey", linewidth=1.2, label="Actual", zorder=1
    )

    if len(preds) == 1:
        # Single prediction
        trial_num = trials.iloc[0]["number"]
        pred_vals = predicted_values[0]
        ax.plot(x, pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Trial {trial_num}", zorder=3)
    else:
        # Multiple predictions
        # Plot all non-best trials
        for i in range(1, len(preds)):  # Skip best (index 0)
            pred_vals = predicted_values[i]
            # Graduated transparency based on ranking
            alpha = 0.2 + (0.3 * (1 - i / len(preds)))
            ax.plot(x, pred_vals, "-", color=color_map[i], linewidth=1.0, alpha=alpha, zorder=2)

        # Add single legend entry for other trials
        ax.plot([], [], "-", color="grey", alpha=0.4, linewidth=1.0, label=f"Trials 2-{len(preds)} (n={len(preds) - 1})")

        # Plot best trial last with markers
        best_trial_num = trials.iloc[0]["number"]
        best_pred_vals = predicted_values[0]
        ax.plot(x, best_pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Best: Trial {best_trial_num}", zorder=4)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=10)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Cases", fontsize=12)

    # Cleaner grid
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax.set_axisbelow(True)  # Grid behind data

    # Tighter legend
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Add subtle formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "cases_by_month.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_cases_by_month_timeseries(actual, preds, trials, output_dir, color_map, start_year):
    if "cases_by_month_timeseries" not in actual:
        print("[WARN] cases_by_month_timeseries not found in calib targets. Skipping plot.")
        return

    # Extract predicted values
    predicted_values = [pred[0]["cases_by_month_timeseries"] for pred in preds]
    actual_values = actual["cases_by_month_timeseries"]

    # Create date range for x-axis
    n_months = len(actual_values)
    months_series = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq="MS")

    # Convert to numeric positions for bar plotting
    x = np.arange(len(months_series))

    fig, ax = plt.subplots(figsize=(12, 6))  # Wider for time series
    ax.set_title("Cases by Month Timeseries", fontsize=14, fontweight="bold")

    # Plot actual data as grey bars (consistent with other plots)
    _bars = ax.bar(
        x, actual_values, width=0.7, color="lightgrey", alpha=0.6, edgecolor="lightgrey", linewidth=1.2, label="Actual", zorder=1
    )

    if len(preds) == 1:
        # Single prediction
        trial_num = trials.iloc[0]["number"]
        pred_vals = predicted_values[0]
        ax.plot(x, pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Trial {trial_num}", zorder=3)
    else:
        # Multiple predictions
        # Plot all non-best trials without markers
        for i in range(1, len(preds)):  # Skip best (index 0)
            pred_vals = predicted_values[i]
            # Graduated transparency based on ranking
            alpha = 0.2 + (0.3 * (1 - i / len(preds)))
            ax.plot(x, pred_vals, "-", color=color_map[i], linewidth=1.0, alpha=alpha, zorder=2)

        # Add single legend entry for other trials
        ax.plot([], [], "-", color="grey", alpha=0.4, linewidth=1.0, label=f"Trials 2-{len(preds)} (n={len(preds) - 1})")

        # Plot best trial last with markers
        best_trial_num = trials.iloc[0]["number"]
        best_pred_vals = predicted_values[0]
        ax.plot(x, best_pred_vals, "-", color=color_map[0], linewidth=2, markersize=7, label=f"Best: Trial {best_trial_num}", zorder=4)

    # Formatting - use subset of dates for cleaner x-axis
    # Show every nth month to avoid crowding
    n_ticks = 12  # Show approximately 12 ticks
    step = max(1, len(months_series) // n_ticks)
    tick_positions = x[::step]
    tick_labels = [months_series[i].strftime("%Y-%m") for i in range(0, len(months_series), step)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cases", fontsize=12)

    # Cleaner grid
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax.set_axisbelow(True)  # Grid behind data

    # Tighter legend
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Add subtle formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "cases_by_month_timeseries.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_cases_by_region(actual, preds, trials, output_dir, color_map):
    if "cases_by_region" not in actual:
        print("[WARN] cases_by_region not found in calib targets. Skipping plot.")
        return

    # Extract predicted values
    predicted_values = [pred[0]["cases_by_region"] for pred in preds]
    actual_value = actual["cases_by_region"]

    region_labels = list(actual_value.keys())
    x = np.arange(len(region_labels))
    actual_vals = list(actual_value.values())

    fig, ax = plt.subplots(figsize=(12, 8))  # Taller for region names
    ax.set_title("Regional Cases", fontsize=14, fontweight="bold")

    # Plot actual data as grey bars
    _bars = ax.bar(x, actual_vals, width=0.7, color="lightgrey", alpha=0.6, edgecolor="lightgrey", linewidth=1.2, label="Actual", zorder=1)

    if len(preds) == 1:
        # Single prediction
        trial_num = trials.iloc[0]["number"]
        pred_vals = list(predicted_values[0].values())
        ax.plot(x, pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Trial {trial_num}", zorder=3)
    else:
        # Multiple predictions
        # Plot all non-best trials without markers
        for i in range(1, len(preds)):  # Skip best (index 0)
            pred_vals = list(predicted_values[i].values())
            # Graduated transparency based on ranking
            alpha = 0.2 + (0.3 * (1 - i / len(preds)))
            ax.plot(x, pred_vals, "-", color=color_map[i], linewidth=1.0, alpha=alpha, zorder=2)

        # Add single legend entry for other trials
        ax.plot([], [], "-", color="grey", alpha=0.4, linewidth=1.0, label=f"Trials 2-{len(preds)} (n={len(preds) - 1})")

        # Plot best trial last with markers
        best_trial_num = trials.iloc[0]["number"]
        best_pred_vals = list(predicted_values[0].values())
        ax.plot(x, best_pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Best: Trial {best_trial_num}", zorder=4)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Cases", fontsize=12)

    # Cleaner grid
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax.set_axisbelow(True)  # Grid behind data

    # Tighter legend
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Add subtle formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "cases_by_region.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_cases_by_region_period(actual, preds, trials, output_dir, color_map):
    if "cases_by_region_period" not in actual:
        print("[WARN] cases_by_region_period not found in calib targets. Skipping plot.")
        return

    region_period_data = actual["cases_by_region_period"]
    predicted_values = [pred[0]["cases_by_region_period"] for pred in preds]

    # Extract all unique periods from all regions
    all_periods = set()
    for region_dict in region_period_data.values():
        all_periods.update(region_dict.keys())
    periods = sorted(all_periods)

    # Extract all unique regions
    regions = sorted(region_period_data.keys())

    # Create figure with subplots stacked vertically (one per period)
    n_periods = len(periods)
    if n_periods == 0:
        return

    fig, axes = plt.subplots(n_periods, 1, figsize=(12, 4 * n_periods))
    if n_periods == 1:
        axes = [axes]

    for period_idx, period in enumerate(periods):
        ax = axes[period_idx]

        # Extract data for this period across all regions
        x = np.arange(len(regions))
        actual_vals = [region_period_data.get(region, {}).get(period, 0) for region in regions]

        # Plot actual data as grey bars
        ax.bar(x, actual_vals, width=0.7, color="lightgrey", alpha=0.6, edgecolor="lightgrey", linewidth=1.2, label="Actual", zorder=1)

        if len(preds) == 1:
            # Single prediction
            trial_num = trials.iloc[0]["number"]
            rep_data = predicted_values[0]
            pred_vals = [rep_data.get(region, {}).get(period, 0) for region in regions]
            ax.plot(x, pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Trial {trial_num}", zorder=3)
        else:
            # Multiple predictions
            # Plot all non-best trials without markers
            for i in range(1, len(preds)):  # Skip best (index 0)
                rep_data = predicted_values[i]
                pred_vals = [rep_data.get(region, {}).get(period, 0) for region in regions]
                # Graduated transparency based on ranking
                alpha = 0.2 + (0.3 * (1 - i / len(preds)))
                ax.plot(x, pred_vals, "-", color=color_map[i], linewidth=1.0, alpha=alpha, zorder=2)

            # Add single legend entry for other trials (only for first subplot)
            if period_idx == 0:
                ax.plot([], [], "-", color="grey", alpha=0.4, linewidth=1.0, label=f"Trials 2-{len(preds)} (n={len(preds) - 1})")

            # Plot best trial last with markers
            best_trial_num = trials.iloc[0]["number"]
            best_rep_data = predicted_values[0]
            best_pred_vals = [best_rep_data.get(region, {}).get(period, 0) for region in regions]
            ax.plot(
                x, best_pred_vals, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Best: Trial {best_trial_num}", zorder=4
            )

        # Formatting
        ax.set_title(f"Regional Cases - {period}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Cases", fontsize=11)

        # Cleaner grid
        ax.grid(True, alpha=0.2, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        # Legend only on first subplot to avoid repetition
        if period_idx == 0:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

        # Add subtle formatting
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "cases_by_region_period.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_cases_by_region_month(actual, preds, trials, output_dir, color_map, start_year):
    if "cases_by_region_month" not in actual:
        print("[WARN] cases_by_region_month not found in calib targets. Skipping plot.")
        return

    cases_by_region_month_actual = actual["cases_by_region_month"]
    predicted_values = [pred[0]["cases_by_region_month"] for pred in preds]
    regions = list(cases_by_region_month_actual.keys())
    n_regions = len(regions)

    if n_regions == 0:
        return

    # Create subplot grid
    n_cols = 2
    n_rows = (n_regions + n_cols - 1) // n_cols  # Ceiling division

    # Define dynamic figure size
    fig_height_per_row = 3.5
    fig_width = 15
    fig_height = n_rows * fig_height_per_row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.suptitle("Regional Monthly Timeseries Comparison", fontsize=16, fontweight="bold")

    # Normalize axes to always be a flat list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Get time series info
    first_key = next(iter(cases_by_region_month_actual))
    n_months = len(cases_by_region_month_actual[first_key])
    months_series = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq="MS")
    x = np.arange(len(months_series))

    for idx, region in enumerate(regions):
        ax = axes[idx]
        actual_timeseries = cases_by_region_month_actual[region]

        # Plot actual data as grey bars
        ax.bar(
            x, actual_timeseries, width=0.7, color="lightgrey", alpha=0.6, edgecolor="lightgrey", linewidth=1.2, label="Actual", zorder=1
        )

        if len(preds) == 1:
            # Single prediction
            trial_num = trials.iloc[0]["number"]
            if region in predicted_values[0]:
                pred_timeseries = predicted_values[0][region]
                ax.plot(x, pred_timeseries, "o-", color=color_map[0], linewidth=2.5, markersize=5, label=f"Trial {trial_num}", zorder=3)
        else:
            # Multiple predictions
            # Plot all non-best trials without markers
            for i in range(1, len(preds)):
                if region in predicted_values[i]:
                    pred_timeseries = predicted_values[i][region]
                    alpha = 0.2 + (0.3 * (1 - i / len(preds)))
                    ax.plot(x, pred_timeseries, "-", color=color_map[i], linewidth=1.0, alpha=alpha, zorder=2)

            # Add single legend entry for other trials (only in first subplot)
            if idx == 0:
                ax.plot([], [], "-", color="grey", alpha=0.4, linewidth=1.0, label=f"Trials 2-{len(preds)} (n={len(preds) - 1})")

            # Plot best trial last with markers
            best_trial_num = trials.iloc[0]["number"]
            if region in predicted_values[0]:
                best_timeseries = predicted_values[0][region]
                ax.plot(
                    x,
                    best_timeseries,
                    "o-",
                    color=color_map[0],
                    linewidth=2.5,
                    markersize=5,
                    label=f"Best: Trial {best_trial_num}",
                    zorder=4,
                )

        # Formatting
        ax.set_title(f"{region.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cases", fontsize=10)

        # Format x-axis with subset of dates
        n_ticks = 6  # Show fewer ticks per subplot
        step = max(1, len(months_series) // n_ticks)
        tick_positions = x[::step]
        tick_labels = [months_series[i].strftime("%Y-%m") for i in range(0, len(months_series), step)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

        # Cleaner grid
        ax.grid(True, alpha=0.2, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        # Remove spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Only add legend to first subplot
        if idx == 0:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Hide any unused subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "cases_by_region_month.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_case_bins_by_region(actual, preds, trials, output_dir, color_map, model_config):
    if "case_bins_by_region" not in actual:
        print("[WARN] case_bins_by_region not found in calib targets. Skipping plot.")
        return

    # Read bin configuration from model config
    bin_config = model_config.get("summary_config", {}).get("case_bins", {})
    bin_labels = bin_config.get("bin_labels", ["0", "1", "2", "3", "4", "5-9", "10-19", "20+"])

    case_bins_by_region_actual = actual["case_bins_by_region"]
    predicted_values = [pred[0]["case_bins_by_region"] for pred in preds]
    regions = list(case_bins_by_region_actual.keys())
    n_regions = len(regions)

    if n_regions == 0:
        return

    # Create subplot grid
    n_cols = 2
    n_rows = (n_regions + n_cols - 1) // n_cols

    fig_height_per_row = 3
    fig_width = 15
    fig_height = n_rows * fig_height_per_row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.suptitle("District Case Count Distribution by Region", fontsize=16, fontweight="bold")

    # Normalize axes to always be a flat list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Get consistent y-axis scale across all subplots
    all_counts = []
    all_counts.extend(case_bins_by_region_actual.values())
    for pred_dict in predicted_values:
        if pred_dict:
            all_counts.extend(pred_dict.values())
    max_count = max(max(counts) if counts else 0 for counts in all_counts)

    for idx, region in enumerate(regions):
        ax = axes[idx]
        actual_counts = case_bins_by_region_actual[region]
        x_positions = np.arange(len(bin_labels))

        # Plot actual data as grey bars
        bars = ax.bar(
            x_positions,
            actual_counts,
            width=0.7,
            color="lightgrey",
            alpha=0.6,
            edgecolor="lightgrey",
            linewidth=1.2,
            label="Actual",
            zorder=1,
        )

        if len(preds) == 1:
            # Single prediction
            trial_num = trials.iloc[0]["number"]
            if region in predicted_values[0]:
                pred_counts = predicted_values[0][region]
                ax.plot(
                    x_positions, pred_counts, "o-", color=color_map[0], linewidth=2.5, markersize=7, label=f"Trial {trial_num}", zorder=3
                )
        else:
            # Multiple predictions
            # Plot all non-best trials without markers
            for i in range(1, len(preds)):
                if region in predicted_values[i]:
                    pred_counts = predicted_values[i][region]
                    alpha = 0.2 + (0.3 * (1 - i / len(preds)))
                    ax.plot(x_positions, pred_counts, "-", color=color_map[i], linewidth=1.0, alpha=alpha, zorder=2)

            # Add single legend entry for other trials (only in first subplot)
            if idx == 0:
                ax.plot([], [], "-", color="grey", alpha=0.4, linewidth=1.0, label=f"Trials 2-{len(preds)} (n={len(preds) - 1})")

            # Plot best trial last with markers
            best_trial_num = trials.iloc[0]["number"]
            if region in predicted_values[0]:
                best_counts = predicted_values[0][region]
                ax.plot(
                    x_positions,
                    best_counts,
                    "-",
                    color=color_map[0],
                    linewidth=2.5,
                    markersize=7,
                    label=f"Best: Trial {best_trial_num}",
                    zorder=4,
                )

        # Formatting
        ax.set_title(f"{region.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Cases", fontsize=10)
        ax.set_ylabel("Number of Districts", fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bin_labels, rotation=0, fontsize=9)
        ax.set_ylim(0, max_count * 1.1)

        # Cleaner grid
        ax.grid(True, alpha=0.2, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        # Remove spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add count annotations on actual bars (optional - can remove if too cluttered)
        for _i, (bar, count) in enumerate(zip(bars, actual_counts, strict=False)):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    count + max_count * 0.02,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="grey",
                )

        # Only add legend to first subplot
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Hide any unused subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "case_bins_by_region.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_case_diff_choropleth_multi(actual, preds, trials, output_dir, color_map, shp, model_config):
    if shp is None or "cases_by_region" not in actual:
        print("[WARN] Cannot create choropleth - missing shapefile or cases_by_region data.")
        return

    predicted_values = [pred[0]["cases_by_region"] for pred in preds]
    actual_cases = actual["cases_by_region"]

    if len(preds) == 1:
        # Single prediction - plot the difference
        plot_case_diff_choropleth(
            shp=shp,
            actual_cases=actual_cases,
            pred_cases=predicted_values[0],
            output_path=output_dir / "case_diff_choropleth.png",
            title=f"Case Count Difference (Actual - Predicted) - Trial {trials.iloc[0]['number']}",
        )
    else:
        # Multiple predictions - create subplot with best, mean, and std
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, wspace=0.15)

        # Calculate statistics across all predictions
        regions = list(actual_cases.keys())
        pred_array = np.array([[pred_dict.get(r, 0) for r in regions] for pred_dict in predicted_values])
        mean_pred = dict(zip(regions, np.mean(pred_array, axis=0), strict=False))
        std_pred = dict(zip(regions, np.std(pred_array, axis=0), strict=False))
        best_pred = predicted_values[0]  # First is best

        # Panel 1: Best prediction difference
        ax1 = fig.add_subplot(gs[0])
        plot_choropleth_panel(
            ax=ax1,
            shp=shp,
            values={r: actual_cases.get(r, 0) - best_pred.get(r, 0) for r in regions},
            title=f"Best Trial ({trials.iloc[0]['number']})",
            cmap="RdBu",
            center_zero=True,
        )

        # Panel 2: Mean prediction difference
        ax2 = fig.add_subplot(gs[1])
        plot_choropleth_panel(
            ax=ax2,
            shp=shp,
            values={r: actual_cases.get(r, 0) - mean_pred.get(r, 0) for r in regions},
            title=f"Mean of {len(preds)} Trials",
            cmap="RdBu",
            center_zero=True,
        )

        # Panel 3: Standard deviation of predictions
        ax3 = fig.add_subplot(gs[2])
        plot_choropleth_panel(ax=ax3, shp=shp, values=std_pred, title="Std Dev of Predictions", cmap="YlOrRd", center_zero=False)

        fig.suptitle("Case Count Differences by Region", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "case_diff_choropleth_multi.png", bbox_inches="tight", dpi=150)
        plt.close()


def plot_choropleth_panel(ax, shp, values, title, cmap="RdBu", center_zero=True):
    """Helper function to plot a single choropleth panel."""
    shp_copy = shp.copy()
    shp_copy["value"] = shp_copy["region"].map(values)

    if center_zero:
        # Center colormap at zero for differences
        max_abs = max(abs(min(values.values())), abs(max(values.values())))
        vmin, vmax = -max_abs, max_abs
    else:
        # Use full range for std dev
        vmin, vmax = 0, max(values.values())

    shp_copy.plot(column="value", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, legend=True, legend_kwds={"shrink": 0.8, "label": title})

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


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


# def plot_top_trials(study, output_dir, n_best=10, title="Top Calibration Results", shp=None, node_lookup=None, start_year=2018):
#     """
#     Plot the top n best calibration trials using the same visualizations as plot_targets.

#     Args:
#         study (optuna.Study): The Optuna study containing trials
#         output_dir (Path): Directory to save plots
#         n_best (int): Number of best trials to plot
#         title (str): Title for the plot
#         shp (GeoDataFrame, optional): Shapefile for choropleth plots
#         node_lookup (dict, optional): Dictionary mapping dot_names to administrative regions
#     """
#     # Get trials sorted by value (ascending)
#     trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
#     top_trials = trials[:n_best]

#     # Load metadata and model config
#     metadata_path = Path(output_dir) / "study_metadata.json"
#     if not metadata_path.exists():
#         raise FileNotFoundError(f"study_metadata.json not found at {metadata_path}")
#     with open(metadata_path) as f:
#         metadata = json.load(f)
#     model_config = metadata.get("model_config", {})

#     # Generate shapefile if not provided
#     if shp is None:
#         try:
#             shp, node_lookup = get_shapefile_from_config(model_config)
#             print("[INFO] Generated shapefile from model config")
#         except Exception as e:
#             print(f"[WARN] Could not generate shapefile: {e}")
#             shp = None

#     # Define consistent colors for trials
#     cmap = cm.get_cmap("tab20")
#     color_map = {f"Trial {trial.number}": cmap(i) for i, trial in enumerate(top_trials)}

#     # Create output directory for top trials plots
#     top_trials_dir = Path(output_dir) / "top_10_trial_plots"
#     top_trials_dir.mkdir(exist_ok=True)

#     # Get actual data from first trial (should be same for all)
#     actual = top_trials[0].user_attrs["actual"]

#     # Total Infected
#     if "total_infected" in actual:
#         plt.figure()
#         plt.title(f"Total Infected - Top {n_best} Trials")
#         width = 0.8 / (n_best + 1)  # Adjust bar width based on number of trials
#         x = np.arange(2)  # Just two bars: Actual and Predicted
#         plt.bar(x[0], actual["total_infected"][0], width, label="Actual", color="black")
#         for i, trial in enumerate(top_trials):
#             pred = trial.user_attrs["predicted"][0]  # Get first replicate
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             plt.bar(
#                 x[1] + (i - n_best / 2) * width, pred["total_infected"][0], width, label=label, color=color_map[f"Trial {trial.number}"]
#             )
#         plt.xticks(x, ["Actual", "Predicted"])
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "total_infected_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Yearly Cases
#     if "yearly_cases" in actual:
#         years = list(range(start_year, start_year + len(actual["yearly_cases"])))
#         plt.figure(figsize=(10, 6))
#         plt.title(f"Yearly Cases - Top {n_best} Trials")
#         plt.plot(years, actual["yearly_cases"], "o-", label="Actual", color="black", linewidth=2)
#         for trial in top_trials:
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             plt.plot(years, pred["yearly_cases"], "o--", label=label, color=color_map[f"Trial {trial.number}"])
#         plt.xlabel("Year")
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "yearly_cases_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Monthly Cases
#     if "monthly_cases" in actual:
#         months = list(range(1, 1 + len(actual["monthly_cases"])))
#         plt.figure(figsize=(10, 6))
#         plt.title(f"Monthly Cases - Top {n_best} Trials")
#         plt.plot(months, actual["monthly_cases"], "o-", label="Actual", color="black", linewidth=2)
#         for trial in top_trials:
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             plt.plot(months, pred["monthly_cases"], "o--", label=label, color=color_map[f"Trial {trial.number}"])
#         plt.xlabel("Month")
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "monthly_cases_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Monthly Timeseries
#     if "monthly_timeseries" in actual:
#         n_months = len(actual["monthly_timeseries"])
#         months_series = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq="MS")
#         plt.figure(figsize=(10, 6))
#         plt.title(f"Monthly Timeseries - Top {n_best} Trials")
#         plt.plot(months_series, actual["monthly_timeseries"], "o-", label="Actual", color="black", linewidth=2)
#         for trial in top_trials:
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             plt.plot(months_series, pred["monthly_timeseries"], "o--", label=label, color=color_map[f"Trial {trial.number}"])
#         plt.xlabel("Month")
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "monthly_timeseries_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Total by Period if available
#     total_by_period_actual = actual.get("total_by_period")
#     if total_by_period_actual:
#         # Use the keys from the dictionary in their natural order
#         period_labels = list(actual["total_by_period"].keys())
#         x = np.arange(len(period_labels))
#         actual_vals = [actual["total_by_period"][period] for period in period_labels]

#         plt.figure(figsize=(10, 6))
#         plt.title(f"Total Cases by Period - Top {n_best} Trials")
#         plt.bar(x, actual_vals, width=0.6, edgecolor="black", facecolor="none", linewidth=1.5, label="Actual")

#         for trial in top_trials:
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             pred_vals = [pred["total_by_period"].get(period, 0) for period in period_labels]
#             plt.scatter(x, pred_vals, label=label, color=color_map[f"Trial {trial.number}"], marker="o", s=50)

#         plt.xticks(x, period_labels, rotation=45, ha="right")
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "total_by_period_comparison.png", bbox_inches="tight")
#         plt.close()

#     # ADM0 Cases if available
#     adm0_actual = actual.get("adm0_cases")
#     if adm0_actual:
#         adm_labels = sorted(actual["adm0_cases"].keys())
#         x = np.arange(len(adm_labels))
#         actual_vals = [actual["adm0_cases"].get(adm, 0) for adm in adm_labels]

#         plt.figure(figsize=(12, 6))
#         plt.title(f"ADM0 Cases - Top {n_best} Trials")
#         plt.bar(x, actual_vals, width=0.6, edgecolor="gray", facecolor="none", linewidth=1.5, label="Actual")

#         for trial in top_trials:
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             pred_vals = [pred["adm0_cases"].get(adm, 0) for adm in adm_labels]
#             plt.scatter(x, pred_vals, label=label, color=color_map[f"Trial {trial.number}"], marker="o", s=50)

#         plt.xticks(x, adm_labels, rotation=45, ha="right")
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "adm0_cases_comparison.png", bbox_inches="tight")
#         plt.close()

#     # ADM01 Cases if available
#     adm01_actual = actual.get("adm01_cases")
#     if adm01_actual:
#         adm_labels = sorted(actual["adm01_cases"].keys())
#         x = np.arange(len(adm_labels))
#         actual_vals = [actual["adm01_cases"].get(adm, 0) for adm in adm_labels]

#         plt.figure(figsize=(12, 6))
#         plt.title(f"ADM01 Regional Cases - Top {n_best} Trials")
#         plt.bar(x, actual_vals, width=0.6, edgecolor="black", facecolor="none", linewidth=1.5, label="Actual")

#         for trial in top_trials:
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             pred_vals = [pred["adm01_cases"].get(adm, 0) for adm in adm_labels]
#             plt.scatter(x, pred_vals, label=label, color=color_map[f"Trial {trial.number}"], marker="o", s=50)

#         plt.xticks(x, adm_labels, rotation=45, ha="right")
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "adm01_cases_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Regional Cases
#     if "regional_cases" in actual:
#         region_labels = list(model_config.get("summary_config", {}).get("region_groups", {}).keys())
#         x = np.arange(len(region_labels))
#         width = 0.8 / (n_best + 1)

#         plt.figure(figsize=(12, 6))
#         plt.title(f"Regional Cases - Top {n_best} Trials")
#         plt.bar(x, actual["regional_cases"], width, label="Actual", color="black")

#         for i, trial in enumerate(top_trials):
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             plt.bar(x + (i + 1) * width, pred["regional_cases"], width, label=label, color=color_map[f"Trial {trial.number}"])

#         plt.xticks(x + width * (n_best / 2), region_labels)
#         plt.ylabel("Cases")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "regional_cases_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Total Nodes with Cases
#     if "nodes_with_cases_total" in actual:
#         plt.figure()
#         plt.title(f"Total Nodes with Cases - Top {n_best} Trials")
#         width = 0.8 / (n_best + 1)
#         x = np.arange(2)  # Just two categories: Actual and Predicted
#         plt.bar(x[0], actual["nodes_with_cases_total"][0], width, label="Actual", color="black")
#         for i, trial in enumerate(top_trials):
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             plt.bar(
#                 x[1] + (i - n_best / 2) * width,
#                 pred["nodes_with_cases_total"][0],
#                 width,
#                 label=label,
#                 color=color_map[f"Trial {trial.number}"],
#             )
#         plt.xticks(x, ["Actual", "Predicted"])
#         plt.ylabel("Nodes")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "nodes_with_cases_total_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Monthly Nodes with Cases
#     if "nodes_with_cases_timeseries" in actual:
#         n_months = len(actual["nodes_with_cases_timeseries"])
#         months = list(range(1, n_months + 1))
#         plt.figure(figsize=(10, 6))
#         plt.title(f"Monthly Nodes with Cases - Top {n_best} Trials")
#         plt.plot(months, actual["nodes_with_cases_timeseries"], "o-", label="Actual", color="black", linewidth=2)
#         for trial in top_trials:
#             pred = trial.user_attrs["predicted"][0]
#             label = f"Trial {trial.number} (value={trial.value:.2f})"
#             plt.plot(months, pred["nodes_with_cases_timeseries"], "o-", label=label, color=color_map[f"Trial {trial.number}"])
#         plt.xlabel("Month")
#         plt.ylabel("Number of Nodes with ≥1 Case")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         plt.savefig(top_trials_dir / "nodes_with_cases_timeseries_comparison.png", bbox_inches="tight")
#         plt.close()

#     # Plot choropleth of case count differences for all trials in one figure
#     if shp is not None and node_lookup is not None:
#         actual = top_trials[0].user_attrs["actual"]
#         if "adm01_cases" in actual:
#             trial_predictions = [(trial.number, trial.value, trial.user_attrs["predicted"][0]["adm01_cases"]) for trial in top_trials]
#             plot_multiple_choropleths(
#                 shp=shp,
#                 node_lookup=node_lookup,
#                 actual_cases=actual["adm01_cases"],
#                 trial_predictions=trial_predictions,
#                 output_path=top_trials_dir / "case_diff_choropleths.png",
#                 legend_position="bottom",  # Add parameter to control legend position
#             )

#         # Plot temporal choropleth for the best trial
#         if "cases_by_region_period" in actual:
#             best_trial = top_trials[0]
#             best_pred = best_trial.user_attrs["predicted"][0]
#             plot_case_diff_choropleth_temporal(
#                 shp=shp,
#                 actual_cases_by_period=actual["cases_by_region_period"],
#                 pred_cases_by_period=best_pred["cases_by_region_period"],
#                 output_path=top_trials_dir / "case_diff_choropleth_temporal_best.png",
#                 title=f"Case Count Difference by Period - Best Trial {best_trial.number} (value={best_trial.value:.2f})",
#             )


def plot_likelihoods_vs_params(study, output_dir=None, use_log=True, figsize=(12, 8), point_size=20, alpha=0.7):
    # Load trials dataframe
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    params = [c for c in df.columns if c.startswith("params_")]
    cols_to_keep = params + ["user_attrs_likelihoods"]  # noqa: RUF005
    df = df[cols_to_keep]
    # Expand dicts into separate columns
    expanded = df["user_attrs_likelihoods"].apply(pd.Series)
    if 0 in expanded.columns:
        expanded = expanded.drop(columns=[0])
    # Optionally prefix the new columns
    expanded = expanded.add_prefix("ll_")
    ll_cols = [c for c in expanded.columns if c.startswith("ll_")]
    # Join back to the original dataframe (and drop the old dict column if you want)
    df_expanded = pd.concat([df.drop(columns=["user_attrs_likelihoods"]), expanded], axis=1)
    # Drop rows where any column is NaN
    df_expanded = df_expanded.dropna(how="any")

    for param in params:
        n = len(ll_cols)
        ncols = 3 if n >= 3 else n
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Shared X data
        x = df_expanded[param].values

        for i, ycol in enumerate(ll_cols):
            ax = axes[i]
            y = df_expanded[ycol].values
            ax.scatter(x, y, s=point_size, alpha=alpha)
            if use_log:
                # Only set log if all positive
                if np.all(y > 0):
                    ax.set_yscale("log")
            like_label = ycol.replace("like_", "")
            param_label = param.replace("params_", "")
            ax.set_title(f"{like_label}")  # vs {param_label}
            ax.set_xlabel(param_label)
            ax.set_ylabel(like_label)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        out = Path(output_dir / "likelihoods_vs_params" / f"likelihoods_vs_{param_label}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
