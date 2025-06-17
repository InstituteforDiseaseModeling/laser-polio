import json
import os
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import sciris as sc
import yaml
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet

import laser_polio as lp

__all__ = ["run_sim"]


if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))


def run_sim(
    config=None,
    init_pop_file=None,
    verbose=1,
    run=True,
    save_init_pop=False,
    save_final_pop=False,
    plot_pars=False,
    use_pim_scalars=False,
    **kwargs,
):
    """
    Set up simulation from config file (YAML + overrides) or kwargs.

    Parameters:
        config (dict): Configuration dictionary with simulation parameters.
        init_pop_file (str): Path to initial population file.
        verbose (int): Verbosity level for logging.
        run (bool): Whether to run the simulation.
        save_init_pop (bool): Whether to save the initial population file.
        save_final_pop (bool): Whether to save the final population file.
        plot_pars (bool): Whether to plot the parameters.
        use_pim_scalars (bool): Whether to use R0 scalars based on polio immunity mapper (PIM) random effects or under weight fraction.
        kwargs (dict): Additional parameters to override in the config.

    Example usage:
        # Use kwargs
        run_sim(regions=["ZAMFARA"], r0=16)

        # Pass in configs directly (or from a file)
        config={"dur": 365 * 2, "gravity_k": 2.2}
        run_sim(config)

        # From command line:
        python -m laser_polio.run_sim --extra-pars='{"gravity_k": 2.2, "r0": 14}'

    """

    config = config or {}
    configs = sc.mergedicts(config, kwargs)

    # Extract simulation setup parameters with defaults or overrides
    regions = configs.pop("regions", ["ZAMFARA"])
    start_year = configs.pop("start_year", 2018)
    n_days = configs.pop("n_days", 365)
    pop_scale = configs.pop("pop_scale", 1)
    init_region = configs.pop("init_region", "ANKA")
    init_prev = configs.pop("init_prev", 0.01)
    results_path = configs.pop("results_path", "results/demo")
    save_plots = configs.pop("save_plots", False)
    save_data = configs.pop("save_data", False)
    plot_pars = configs.pop("plot_pars", plot_pars)
    init_pop_file = configs.pop("init_pop_file", init_pop_file)
    background_seeding = configs.pop("background_seeding", False)
    background_seeding_freq = configs.pop("background_seeding_freq", 30)
    background_seeding_node_frac = configs.pop("background_seeding_node_frac", 0.3)
    background_seeding_prev = configs.pop("background_seeding_prev", 0.0001)
    use_pim_scalars = configs.pop("use_pim_scalars", use_pim_scalars)
    init_immun_scalar = configs.pop("init_immun_scalar", 1.0)
    r0_scalar_wt_slope = configs.pop("r0_scalar_wt_slope", 24)
    r0_scalar_wt_intercept = configs.pop("r0_scalar_wt_intercept", 0.2)

    # Geography
    dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv", verbose=verbose)
    node_lookup = lp.get_node_lookup(lp.root / "data/node_lookup.json", dot_names)
    # dist_matrix = lp.get_distance_matrix(lp.root / "data/distance_matrix_africa_adm2.h5", dot_names)
    shp = gpd.read_file(filename=lp.root / "data/shp_africa_low_res.gpkg", layer="adm2")
    shp = shp[shp["dot_name"].isin(dot_names)]
    # Sort the GeoDataFrame by the order of dot_names
    shp.set_index("dot_name", inplace=True)
    shp = shp.loc[dot_names].reset_index()

    # Immunity
    init_immun = pd.read_hdf(lp.root / "data/init_immunity_0.5coverage_january.h5", key="immunity")
    init_immun = init_immun.set_index("dot_name").loc[dot_names]
    init_immun = init_immun[init_immun["period"] == start_year]
    # Apply scalar multiplier to immunity values, clipping to [0.0, 1.0]
    immunity_cols = [col for col in init_immun.columns if col.startswith("immunity_")]
    init_immun[immunity_cols] = init_immun[immunity_cols].clip(lower=0.0, upper=1.0) * init_immun_scalar
    # Apply geographic scalars if specified in configs
    if "immun_scalar_kano" in configs:
        kano_scalar = configs.pop("immun_scalar_kano")
        kano_mask = init_immun.index.str.contains("NIGERIA:KANO")
        init_immun.loc[kano_mask, immunity_cols] *= kano_scalar
    if "immun_scalar_kebbi" in configs:
        kebbi_scalar = configs.pop("immun_scalar_kebbi")
        kebbi_mask = init_immun.index.str.contains("NIGERIA:KEBBI")
        init_immun.loc[kebbi_mask, immunity_cols] *= kebbi_scalar
    init_immun[immunity_cols] = init_immun[immunity_cols].clip(upper=1.0, lower=0.0)

    # Initial infection seeding
    init_prevs = np.zeros(len(dot_names))
    prev_indices = [i for i, dot_name in enumerate(dot_names) if init_region in dot_name]
    if not prev_indices:
        raise ValueError(f"No nodes found containing '{init_region}'")
    init_prevs[prev_indices] = init_prev
    # Make dtype match init_prev type
    if isinstance(init_prev, int):
        init_prevs = init_prevs.astype(int)
    if verbose >= 2:
        print(f"Seeding infection in {len(prev_indices)} nodes at {init_prev:.3f} prevalence.")
    # Set up background seeding if specified
    if background_seeding:
        print("Using background seeding")
        background_seeds = lp.make_background_seeding_schedule(
            node_lookup,
            start_date=lp.date(f"{start_year}-01-01"),
            sim_duration=n_days,
            prevalence=background_seeding_prev,
            fraction_of_nodes=background_seeding_node_frac,
            frequency=background_seeding_freq,
            rng=np.random.default_rng(configs.get("seed", None)),  # Use the seed from configs
        )
        # Merge with existing seed_schedule if it exists
        if "seed_schedule" in configs and configs["seed_schedule"] is not None:
            configs["seed_schedule"] += background_seeds
        else:
            configs["seed_schedule"] = background_seeds

    # SIA schedule
    start_date = lp.date(f"{start_year}-01-01")
    historic = pd.read_csv(lp.root / "data/sia_historic_schedule.csv")
    future = pd.read_csv(lp.root / "data/sia_scenario_1.csv")
    sia_schedule = lp.process_sia_schedule_polio(pd.concat([historic, future]), dot_names, start_date, n_days, filter_to_type2=True)

    # Demographics and risk
    df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
    df_comp = df_comp[df_comp["year"] == start_year]
    pop = df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values * pop_scale
    cbr = df_comp.set_index("dot_name").loc[dot_names, "cbr"].values
    ri = df_comp.set_index("dot_name").loc[dot_names, "ri_eff"].values
    ri_ipv = df_comp.set_index("dot_name").loc[dot_names, "dpt3"].values
    # SIA probabilities
    sia_re = df_comp.set_index("dot_name").loc[dot_names, "sia_random_effect"].values
    sia_prob = lp.calc_sia_prob_from_rand_eff(sia_re, center=0.5, scale=1.0)
    # R0 scalars
    underwt = df_comp.set_index("dot_name").loc[dot_names, "prop_underwt"].values
    r0_scalars_wt = (
        1 / (1 + np.exp(r0_scalar_wt_slope * (0.22 - underwt)))
    ) + r0_scalar_wt_intercept  # The 0.22 is the mean of Nigeria underwt
    # Scale PIM estimates using Nigeria mins and maxes to keep this consistent with the underweight scaling when geography is not Nigeria
    # TODO: revisit this section if using geography outside Nigeria
    pim_re = df_comp["reff_random_effect"].values  # get all values
    nig_min = -0.0786359245626656
    nig_max = 2.200145038240859
    pim_scaled = (pim_re - nig_min) / (nig_max - nig_min)
    # pim_scaled = (pim_re - pim_re.min()) / (pim_re.max() - pim_re.min())  # Rescale to [0, 1]
    df_comp.loc[:, "pim_scaled"] = pim_scaled
    pim_scaled = df_comp.set_index("dot_name").loc[dot_names, "pim_scaled"].values
    r0_scalar_pim = pim_scaled * (r0_scalars_wt.max() - r0_scalars_wt.min()) + r0_scalars_wt.min()
    if use_pim_scalars:
        r0_scalars = r0_scalar_pim
    else:
        r0_scalars = r0_scalars_wt

    # Validate all arrays match
    assert all(len(arr) == len(dot_names) for arr in [shp, init_immun, node_lookup, init_prevs, pop, cbr, ri, ri_ipv, sia_prob, r0_scalars])

    # Setup results path
    if results_path is None:
        results_path = Path("results/default")  # Provide a default path

    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_path)

    # Base parameters (can be overridden)
    base_pars = {
        "start_date": start_date,
        "dur": n_days,
        "n_ppl": pop,
        "age_pyramid_path": lp.root / "data/Nigeria_age_pyramid_2024.csv",
        "cbr": cbr,
        "init_immun": init_immun,
        "init_prev": init_prevs,
        "r0_scalars": r0_scalars,
        "distances": None,
        "shp": shp,
        "node_lookup": node_lookup,
        "vx_prob_ri": ri,
        "vx_prob_ipv": ri_ipv,
        "sia_schedule": sia_schedule,
        "vx_prob_sia": sia_prob,
        "verbose": verbose,
        "stop_if_no_cases": False,
    }

    # Dynamic values passed by user/CLI/Optuna
    pars = PropertySet({**base_pars, **configs})
    # Plot pars
    if plot_pars:
        lp.plot_pars(pars, shp, results_path)

    def from_file(init_pop_file):
        # logger.info(f"Initializing SEIR_ABM from file: {init_pop_file}")
        people, results_R, pars_loaded = LaserFrame.load_snapshot(init_pop_file)

        sim = lp.SEIR_ABM.init_from_file(people, pars)
        if pars_loaded and "r0" in pars_loaded:
            sim.pars.old_r0 = pars_loaded["r0"]
        disease_state = lp.DiseaseState_ABM.init_from_file(sim)
        vd = lp.VitalDynamics_ABM.init_from_file(sim)
        sia = lp.SIA_ABM.init_from_file(sim)
        ri = lp.RI_ABM.init_from_file(sim)
        tx = lp.Transmission_ABM.init_from_file(sim)
        sim.results.R = results_R
        sim._components = [type(vd), type(disease_state), type(tx), type(ri), type(sia)]
        sim.instances = [vd, disease_state, tx, ri, sia]
        return sim

    def regular():
        sim = lp.SEIR_ABM(pars)
        components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM]
        if pars.vx_prob_ri is not None:
            components.append(lp.RI_ABM)
        if pars.vx_prob_sia is not None:
            components.append(lp.SIA_ABM)
        sim.components = components
        return sim

    # Either initialize the sim from file or create a sim from scratch
    if init_pop_file:
        sim = from_file(init_pop_file)
    else:
        sim = regular()
        if save_init_pop:
            sim.people.save_snapshot(results_path / "init_pop.h5", sim.results.R[:], sim.pars)

    # Safety checks
    if verbose >= 3:
        sc.pp(pars.to_dict())  # Print pars
        print(f"sim.people.count: {sim.people.count}")
        print(f"disease state counts: {np.bincount(sim.people.disease_state[: sim.people.count])}")
        print(f"infected: {np.where(sim.people.disease_state[: sim.people.count] == 2)}")

    # Run sim
    if run:
        sim.run()
        if save_plots:
            Path(results_path).mkdir(parents=True, exist_ok=True)
            sim.plot(save=True, results_path=results_path)
        if save_data:
            Path(results_path).mkdir(parents=True, exist_ok=True)
            lp.save_results_to_csv(sim, filename=Path(results_path) / "simulation_results.csv")
        if save_final_pop:
            sim.people.save_snapshot(results_path / "final_pop.h5", sim.results.R[:], sim.pars)

    return sim


# Add command-line interface (CLI) for running the simulation
@click.command()
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    default=None,
    help="Optional path to base model config YAML",
)
@click.option(
    "--params-file",
    type=click.Path(exists=True),
    default=None,
    help="Optional trial parameter JSON file (Optuna override)",
)
@click.option(
    "--results-path",
    type=str,
    default="output",
    show_default=True,
    help="Path to write simulation results (CSV format)",
)
@click.option(
    "--extra-pars", type=str, default=None, help='Optional JSON string with additional parameters, e.g. \'{"r0": 14.2, "gravity_k": 1.0}\''
)
@click.option(
    "--init-pop-file",
    type=click.Path(exists=True),
    default=None,
    help="Optional initial population file",
)
@click.option(
    "--save-init-pop",
    is_flag=True,
    help="Save initial population file",
)
@click.option(
    "--save-final-pop",
    is_flag=True,
    help="Save final population file",
)
def main(model_config, params_file, results_path, extra_pars, init_pop_file, save_init_pop, save_final_pop):
    """Run polio LASER simulation with optional config and parameter overrides."""

    config = {}

    # Only load model config if user provided the flag
    if model_config:
        with open(model_config) as f:
            config = yaml.safe_load(f)
        print(f"[INFO] Loaded config from {model_config}")
    else:
        print("[INFO] No model config provided; using defaults.")

    # Load parameter overrides if provided
    if params_file:
        with open(params_file) as f:
            optuna_params = json.load(f)
        config.update(optuna_params)
        print(f"[INFO] Loaded Optuna params from {params_file}")

    # Inject result path (always)
    if results_path:
        config["results_path"] = results_path

    if extra_pars:
        config.update(json.loads(extra_pars))

    # Run the sim: save_pop and init_pop_file are mutually exclusive, not yet enforced
    run_sim(config=config, init_pop_file=init_pop_file, save_init_pop=save_init_pop, save_final_pop=save_final_pop)


# ---------------------------- CLI ENTRY ----------------------------
if __name__ == "__main__":
    main()
