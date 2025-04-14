import json
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sciris as sc
import yaml
from laser_core.propertyset import PropertySet

import laser_polio as lp

__all__ = ["setup_sim"]


if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))


def setup_sim(config=None, **kwargs):
    """Set up simulation from config file (YAML + overrides) or kwargs."""
    config = config or {}

    # Extract simulation setup parameters with defaults or overrides
    regions = config.get("regions", kwargs.get("regions", ["NIGERIA"]))
    start_year = config.get("start_year", kwargs.get("start_year", 2019))
    n_days = config.get("n_days", kwargs.get("n_days", 365))
    pop_scale = config.get("pop_scale", kwargs.get("pop_scale", 0.01))
    init_region = config.get("init_region", kwargs.get("init_region", "ANKA"))
    init_prev = float(config.get("init_prev", kwargs.get("init_prev", 0.01)))
    results_path = config.get("results_path", kwargs.get("results_path", "results/demo"))
    actual_data = config.get("actual_data", kwargs.get("actual_data", "data/epi_africa_20250408.h5"))
    save_plots = config.get("save_plots", kwargs.get("save_plots", False))
    save_data = config.get("save_data", kwargs.get("save_data", False))

    # Geography
    dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
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
    print(f"[INFO] Seeding infection in {len(prev_indices)} nodes at {init_prev:.3f} prevalence.")

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
    reff_re = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values
    sia_prob = lp.calc_sia_prob_from_rand_eff(sia_re)
    r0_scalars = lp.calc_r0_scalars_from_rand_eff(reff_re)

    # Validate all arrays match
    assert all(len(arr) == len(dot_names) for arr in [dist_matrix, init_immun, node_lookup, init_prevs, pop, cbr, ri, sia_prob, r0_scalars])

    # Load the actual case data
    epi = lp.get_epi_data(actual_data, dot_names, node_lookup, start_year, n_days)
    epi.rename(columns={"cases": "P"}, inplace=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)
    epi.to_csv(results_path + "/actual_data.csv", index=False)

    # Base parameters (can be overridden)
    pars = PropertySet(
        {
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
            "actual_data": epi,
        }
    )

    # Inject Optuna trial params if any exist
    if Path("params.json").exists():
        with open("params.json") as f:
            optuna_params = json.load(f)
        print("[INFO] Loaded Optuna trial params:", optuna_params)
        pars += optuna_params

    # Run sim
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM, lp.RI_ABM, lp.SIA_ABM]

    # Run simulation
    print("[INFO] Running simulation...")
    sim.run()

    # Save results
    if save_plots:
        sim.plot(save=True, results_path=results_path)
    if save_data:
        Path(results_path).mkdir(parents=True, exist_ok=True)
        lp.save_results_to_csv(sim, filename=results_path + "/simulation_results.csv")

    sc.printcyan("Done.")


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
def main(model_config, params_file, results_path):
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

    # Run the sim
    setup_sim(config=config)


# ---------------------------- CLI ENTRY ----------------------------
if __name__ == "__main__":
    main()
