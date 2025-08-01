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
    print("run_sim started")
    config = config or {}
    configs = sc.mergedicts(config, kwargs)

    # Extract simulation setup parameters with defaults or overrides
    regions = configs.pop("regions", ["ZAMFARA"])
    admin_level = configs.pop("admin_level", None)  # level to match region strings against: None: dot_name, 0: adm0, 1: adm1, 2: adm2
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
    r0_scalar_wt_center = configs.pop("r0_scalar_wt_center", 0.22)
    sia_re_center = configs.pop("sia_re_center", 0.5)
    sia_re_scale = configs.pop("sia_re_scale", 1.0)

    # Geography
    dot_names = lp.find_matching_dot_names(
        regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv", verbose=verbose, admin_level=admin_level
    )
    node_lookup = lp.get_node_lookup(lp.root / "data/node_lookup.json", dot_names)
    shp = gpd.read_file(filename=lp.root / "data/shp_africa_low_res.gpkg", layer="adm2")
    shp = shp[shp["dot_name"].isin(dot_names)]
    # Sort the GeoDataFrame by the order of dot_names
    shp.set_index("dot_name", inplace=True)
    shp = shp.loc[dot_names].reset_index()
    # Check that the ordering is correct
    node_lookup_dot_names = [node_lookup[i]["dot_name"] for i in sorted(node_lookup.keys())]
    assert np.all(node_lookup_dot_names == dot_names), "Node lookup dot names do not match dot names"
    shp_dot_names = shp["dot_name"].tolist()
    assert np.all(shp_dot_names == dot_names), "shp dot names do not match dot names"

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
    pop = (df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values * pop_scale).astype(int)
    cbr = df_comp.set_index("dot_name").loc[dot_names, "cbr"].values
    ri = df_comp.set_index("dot_name").loc[dot_names, "ri_eff"].values
    ri_ipv = df_comp.set_index("dot_name").loc[dot_names, "dpt3"].values
    # SIA probabilities
    sia_re = df_comp.set_index("dot_name").loc[dot_names, "sia_random_effect"].values
    sia_prob = lp.calc_sia_prob_from_rand_eff(sia_re, center=sia_re_center, scale=sia_re_scale)
    # R0 scalars
    underwt = df_comp.set_index("dot_name").loc[dot_names, "prop_underwt"].values
    r0_scalars_wt = (
        1 / (1 + np.exp(r0_scalar_wt_slope * (r0_scalar_wt_center - underwt)))
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

    # --- Calculate the number of initial susceptible people ---

    # Load the age pyramid
    age_pyramid = lp.load_age_pyramid(lp.root / "data/Nigeria_age_pyramid_2024.csv")
    age_pyramid["age_min_months_pyramid"] = age_pyramid["age_min"] * 12  # Convert to months
    age_pyramid["age_max_months_pyramid"] = age_pyramid["age_max"] * 12  # Convert to months
    age_pyramid = age_pyramid.drop(columns=["age_min", "age_max"])
    age_pyramid = age_pyramid.rename(columns={"pop_frac": "pop_frac_pyramid"})

    # Immunity
    init_immun = pd.read_hdf(lp.root / "data/init_immunity_0.5coverage_january.h5", key="immunity")
    init_immun = init_immun.set_index("dot_name").loc[dot_names]
    init_immun = init_immun[init_immun["period"] == start_year]
    # Apply scalar multiplier to immunity values, clipping to [0.0, 1.0]
    immunity_cols = [col for col in init_immun.columns if col.startswith("immunity_")]
    init_immun[immunity_cols] = init_immun[immunity_cols].clip(lower=0.0, upper=1.0) * init_immun_scalar
    # Set immunity for 15+ to 1.0
    init_immun.loc[:, "immunity_180_1200"] = 1.0
    # Wide → Long
    init_immun_long = init_immun.reset_index().melt(
        id_vars="dot_name",
        value_vars=[col for col in init_immun.columns if col.startswith("immunity_")],
        var_name="age_bin",
        value_name="immune_frac",
    )
    # Parse age bins into min/max months
    init_immun_long[["age_min_months_immun", "age_max_months_immun"]] = init_immun_long["age_bin"].str.extract(r"immunity_(\d+)_(\d+)")
    init_immun_long[["age_min_months_immun", "age_max_months_immun"]] = init_immun_long[
        ["age_min_months_immun", "age_max_months_immun"]
    ].astype(int)
    init_immun_long["age_max_months_immun"] += 1  # Make age_max exclusive
    init_immun_long = init_immun_long.drop(columns=["age_bin"])
    # Perform a cross join and filter down to rows where the bins overlap
    # Add temporary join key for cross-join
    init_immun_long["key"] = 1
    age_pyramid["key"] = 1
    # Cross join: all age bins for all pyramid bins
    age_merged = pd.merge(init_immun_long, age_pyramid, on="key").drop("key", axis=1)
    # Filter to overlapping age bins (i.e., where any overlap exists)
    # This logic matches: (start1 < end2) & (start2 < end1)
    age_merged = age_merged[
        (age_merged["age_min_months_immun"] < age_merged["age_max_months_pyramid"])
        & (age_merged["age_max_months_immun"] > age_merged["age_min_months_pyramid"])
    ]
    # Compute the overlap width (in months)
    age_merged["overlap_months"] = (
        np.minimum(age_merged["age_max_months_immun"], age_merged["age_max_months_pyramid"])
        - np.maximum(age_merged["age_min_months_immun"], age_merged["age_min_months_pyramid"])
    ).clip(lower=0)
    # Calculate overlap weight as fraction of the pyramid bin
    age_merged["weight"] = age_merged["overlap_months"] / (age_merged["age_max_months_pyramid"] - age_merged["age_min_months_pyramid"])
    age_merged.drop(
        columns=["pop"], inplace=True
    )  # Drop the pop column since this is for all of Nigeria. We'll replace with node-level total pop below
    # Attach pop data and node id
    node_info = pd.DataFrame(
        {
            "node_id": sorted(node_lookup.keys()),
            "dot_name": dot_names,
            "pop_total": pop,
        }
    )
    age_merged = age_merged.merge(node_info, on="dot_name", how="left")
    # Adjust population count in that bin accordingly
    age_merged["pop_in_age_bin"] = age_merged["pop_total"] * age_merged["pop_frac_pyramid"] * age_merged["weight"]
    # Compute immune/susceptible counts
    age_merged["n_immune"] = age_merged["pop_in_age_bin"] * age_merged["immune_frac"]
    age_merged["n_susceptible"] = age_merged["pop_in_age_bin"] * (1 - age_merged["immune_frac"])
    # Group and summarize
    sus_by_age_node = (
        age_merged.groupby(["dot_name", "node_id", "age_min_months_immun", "age_max_months_immun"])[["n_susceptible", "n_immune"]]
        .sum()
        .round()
        .astype(int)
        .reset_index()
    )
    # Sum by dot_name
    immun_summary = sus_by_age_node.groupby("dot_name")[["n_immune", "n_susceptible"]].sum()
    # Account for rounding errors & handle them in the oldest age bin
    pop_diff = pop - immun_summary["n_immune"] - immun_summary["n_susceptible"]
    sus_by_age_node.loc[sus_by_age_node["age_max_months_immun"] == 1201, "n_immune"] += pop_diff.values
    # Re-calculate the immune & susceptible counts
    immun_summary = sus_by_age_node.groupby("dot_name")[["n_immune", "n_susceptible"]].sum()
    # Convert age_min_months_immun to years
    sus_by_age_node["age_min_yr"] = sus_by_age_node["age_min_months_immun"] / 12
    # Convert age_max_months_immun to years
    sus_by_age_node["age_max_yr"] = sus_by_age_node["age_max_months_immun"] / 12
    # Drop age_min_months_immun and age_max_months_immun
    sus_by_age_node = sus_by_age_node.drop(columns=["age_min_months_immun", "age_max_months_immun"])
    sus_summary = sus_by_age_node.groupby("dot_name")["n_susceptible"].sum().astype(int)
    assert np.all(immun_summary["n_immune"] + immun_summary["n_susceptible"] <= pop), (
        "Immune + susceptible counts are greater than population counts"
    )
    assert np.all(sus_summary <= pop), "Susceptible counts are greater than population counts"

    # ---- Backcalculate RI IPV Protection ----
    # IPV prevents paralysis but does not block transmission.
    # Since IPV and OPV immunity groups are assumed to overlap, and OPV-protected individuals
    # were already marked as Recovered (i.e., immune to both transmission and paralysis),
    # we only need to assign IPV protection to those who are not already immune.
    # Therefore, IPV protection is only applied when IPV coverage exceeds OPV-derived immunity.
    # Initialize IPV protection column
    sus_by_age_node["n_ipv_protected"] = 0
    # Check if IPV parameters are available
    if ri_ipv is not None and len(ri_ipv) > 0:
        # IPV eligibility threshold (must be born after ipv_start_year) + 98 days (roughly the timing of 2nd dose of RI IPV (+ 3rd dose of OPV))
        # Convert to years for comparison with age bins
        ipv_start_year = config.get("ipv_start_year", 2015)  # Default IPV start year is 2015
        max_age_for_ipv_years = start_date.year - ipv_start_year + (98 / 365)
        # Create mapping from dot_name to ri_ipv coverage
        ipv_coverage_map = dict(zip(dot_names, ri_ipv, strict=False))
        # IPV minimum age threshold in years (98 days)
        ipv_min_age_years = 98 / 365
        # Calculate IPV protection for each row in sus_by_age_node
        for idx, row in sus_by_age_node.iterrows():
            dot_name = row["dot_name"]
            age_min_yr = row["age_min_yr"]
            age_max_yr = row["age_max_yr"]
            n_susceptible = row["n_susceptible"]
            n_immune = row["n_immune"]
            # Check if this age bin has any overlap with IPV eligibility
            if age_max_yr >= ipv_min_age_years and age_min_yr <= max_age_for_ipv_years:
                # Get IPV coverage for this node
                vx_prob_ipv = ipv_coverage_map.get(dot_name, 0)
                if vx_prob_ipv > 0:
                    # Calculate total population in this age bin
                    total_pop = n_susceptible + n_immune
                    if total_pop > 0:
                        # Calculate the proportion of this age bin that's eligible for IPV
                        # Eligible age range: [ipv_min_age_years, max_age_for_ipv_years]
                        # Age bin range: [age_min_yr, age_max_yr]

                        # Find the overlap between eligible age range and age bin
                        overlap_min = max(age_min_yr, ipv_min_age_years)
                        overlap_max = min(age_max_yr, max_age_for_ipv_years)

                        if overlap_max > overlap_min:
                            # Calculate eligible fraction within this age bin
                            age_bin_width = age_max_yr - age_min_yr
                            overlap_width = overlap_max - overlap_min
                            eligible_fraction = overlap_width / age_bin_width if age_bin_width > 0 else 0

                            # Current immune fraction in this age bin
                            immune_fraction = n_immune / total_pop

                            # IPV gap: additional protection beyond existing immunity
                            ipv_gap = max(0, vx_prob_ipv - immune_fraction)

                            # Apply IPV protection to eligible portion only
                            # IPV protects against paralysis but not transmission, so these remain susceptible for transmission
                            eligible_pop = total_pop * eligible_fraction
                            eligible_susceptible = n_susceptible * eligible_fraction
                            n_ipv_protected = min(eligible_susceptible, eligible_pop * ipv_gap)
                            sus_by_age_node.loc[idx, "n_ipv_protected"] = int(n_ipv_protected)

    # Validate all arrays match
    assert all(len(arr) == len(dot_names) for arr in [shp, node_lookup, init_prevs, pop, cbr, ri, ri_ipv, sia_prob, r0_scalars])

    # Setup results path
    if results_path is None:
        results_path = Path("results/default")  # Provide a default path

    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_path)

    # Base parameters (can be overridden)
    base_pars = {
        "start_date": start_date,
        "dur": n_days,
        "init_pop": pop,
        "init_sus_by_age": sus_by_age_node,
        "cbr": cbr,
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
        # Experience shows that the current math, even the fudge factor in load_snapshot,
        # is resulting in undersizing the capacity such that we exceed in the last timestep
        # in very large simulations. Easiest approach is to add a week or two to sim length
        # we tell load_snapshot. Yes, we're lying to this function.
        people, results_R, pars_loaded = LaserFrame.load_snapshot(
            init_pop_file, n_ppl=pars["init_pop"], cbr=pars["cbr"], nt=pars["dur"] + 10
        )  # +10 fudge factor

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
        print("Loading initial pop.")
        sim = from_file(init_pop_file)
    else:
        print("Initializing initial pop.")
        sim = regular()
        if save_init_pop:
            sim.people.save_snapshot(results_path / "init_pop.h5", sim.results.R[:], sim.pars)
    print("Initialized")

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

    # Run the sim: save_init_pop and init_pop_file are mutually exclusive, not yet enforced
    run_sim(config=config, init_pop_file=init_pop_file, save_init_pop=save_init_pop, save_final_pop=save_final_pop)


# ---------------------------- CLI ENTRY ----------------------------
if __name__ == "__main__":
    main()
