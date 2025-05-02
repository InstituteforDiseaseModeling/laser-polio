import json
import os
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import sciris as sc
import yaml
from laser_core.propertyset import PropertySet

import laser_polio as lp
from laser_polio.laserframeio import LaserFrameIO

__all__ = ["run_sim"]


if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))


def run_sim(config=None, init_pop_file=None, verbose=1, run=True, save_pop=False, **kwargs):
    """
    Set up simulation from config file (YAML + overrides) or kwargs.

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
    init_prev = float(configs.pop("init_prev", 0.01))
    results_path = configs.pop("results_path", "results/demo")
    actual_data = configs.pop("actual_data", "data/epi_africa_20250421.h5")
    save_plots = configs.pop("save_plots", False)
    save_data = configs.pop("save_data", False)

    # Geography
    dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv", verbose=verbose)
    node_lookup = lp.get_node_lookup("data/node_lookup.json", dot_names)
    dist_matrix = lp.get_distance_matrix(lp.root / "data/distance_matrix_africa_adm2.h5", dot_names)

    # Immunity
    init_immun = pd.read_hdf(lp.root / "data/init_immunity_0.5coverage_january.h5", key="immunity")
    init_immun = init_immun.set_index("dot_name").loc[dot_names]
    init_immun = init_immun[init_immun["period"] == start_year]

    # Initial infection seeding
    init_prevs = np.zeros(len(dot_names))
    prev_indices = [i for i, dot_name in enumerate(dot_names) if init_region in dot_name]
    if not prev_indices:
        raise ValueError(f"No nodes found containing '{init_region}'")
    init_prevs[prev_indices] = init_prev
    if verbose >= 2:
        print(f"Seeding infection in {len(prev_indices)} nodes at {init_prev:.3f} prevalence.")

    # SIA schedule
    start_date = lp.date(f"{start_year}-01-01")
    historic = pd.read_csv(lp.root / "data/sia_historic_schedule.csv")
    future = pd.read_csv(lp.root / "data/sia_scenario_1.csv")
    sia_schedule = lp.process_sia_schedule_polio(pd.concat([historic, future]), dot_names, start_date)

    # Demographics and risk
    df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
    df_comp = df_comp[df_comp["year"] == start_year]
    pop = df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values * pop_scale
    cbr = df_comp.set_index("dot_name").loc[dot_names, "cbr"].values
    ri = df_comp.set_index("dot_name").loc[dot_names, "ri_eff"].values
    sia_re = df_comp.set_index("dot_name").loc[dot_names, "sia_random_effect"].values
    # reff_re = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values
    # TODO Need to REDO random effect probs since they might've been based on the wrong data. Also, need to do the processing before filtering because of the centering & scaling
    sia_prob = lp.calc_sia_prob_from_rand_eff(sia_re)
    # r0_scalars = lp.calc_r0_scalars_from_rand_eff(reff_re)
    # Calcultate geographic R0 modifiers based on underweight data (one for each node)
    underwt = df_comp.set_index("dot_name").loc[dot_names, "prop_underwt"].values
    r0_scalars = (1 / (1 + np.exp(24 * (0.22 - underwt)))) + 0.2  # The 0.22 is the mean of Nigeria underwt
    # # Check Zamfara means
    # print(f"{underwt[-14:]}")
    # print(f"{r0_scalars[-14:]}")

    # Validate all arrays match
    assert all(len(arr) == len(dot_names) for arr in [dist_matrix, init_immun, node_lookup, init_prevs, pop, cbr, ri, sia_prob, r0_scalars])

    # Setup results path
    if results_path is None:
        results_path = Path("results/default")  # Provide a default path

    # Load the actual case data
    epi = lp.get_epi_data(actual_data, dot_names, node_lookup, start_year, n_days)
    epi.rename(columns={"cases": "P"}, inplace=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_path)
    epi.to_csv(results_path / "actual_data.csv", index=False)

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
        "distances": dist_matrix,
        "node_lookup": node_lookup,
        "vx_prob_ri": ri,
        "sia_schedule": sia_schedule,
        "vx_prob_sia": sia_prob,
        "verbose": verbose,
        "stop_if_no_cases": False,
    }

    # Dynamic values passed by user/CLI/Optuna
    pars = PropertySet({**base_pars, **configs})

    # Print pars
    # TODO: make this optional
    # sc.pp(pars.to_dict())

    def from_file(init_pop_file):
        sim = lp.SEIR_ABM.init_from_file(init_pop_file, pars)
        disease_state = lp.DiseaseState_ABM.init_from_file(sim)
        vd = lp.VitalDynamics_ABM.init_from_file(sim)
        sia = lp.SIA_ABM.init_from_file(sim)
        ri = lp.RI_ABM.init_from_file(sim)
        tx = lp.Transmission_ABM.init_from_file(sim)
        sim._components = [type(vd), type(disease_state), type(tx), type(ri), type(sia)]
        sim.instances = [vd, disease_state, tx, ri, sia]
        # reload results.R
        # lots of questionable ad-hod decision-making here for now
        eula_pop_file = init_pop_file.replace("init", "eula")
        if not os.path.exists(eula_pop_file):
            raise ValueError(f"Unable to find required eula pop file: {eula_pop_file}")
        with h5py.File(eula_pop_file, "r") as hdf:
            sim.results.R = hdf["results_R"][:]
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
        if save_pop:
            with h5py.File(results_path / "init_pop.h5", "w") as f:
                LaserFrameIO.save_to_group(sim.people, f.create_group("people"))  # Save the people frame
                f.create_dataset("recovered", data=sim.results.R[:])  # Save the R result array

    # Run sim
    if run:
        sim.run()
        if save_plots:
            Path(results_path).mkdir(parents=True, exist_ok=True)
            sim.plot(save=True, results_path=results_path)
        if save_data:
            Path(results_path).mkdir(parents=True, exist_ok=True)
            lp.save_results_to_csv(sim, filename=results_path / "simulation_results.csv")

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
    default="simulation_results.csv",
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
    help="Optional initial population file (e.g., CSV or JSON format)",
)
def main(model_config, params_file, results_path, extra_pars, init_pop_file):
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

    # Run the sim
    run_sim(config=config, init_pop_file=init_pop_file)


# ---------------------------- CLI ENTRY ----------------------------
if __name__ == "__main__":
    main()
