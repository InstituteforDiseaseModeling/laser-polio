import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
from laser_core.propertyset import PropertySet

import laser_polio as lp

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
    save_plots = config.get("save_plots", kwargs.get("save_plots", False))
    save_data = config.get("save_data", kwargs.get("save_data", False))

    print(f"[INFO] Using init_prev = {init_prev}")

    # 1. Identify regions
    dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

    # 2. Geography
    centroids = pd.read_csv(lp.root / "data/shp_names_africa_adm2.csv").set_index("dot_name").loc[dot_names]
    dist_matrix = lp.get_distance_matrix(lp.root / "data/distance_matrix_africa_adm2.h5", dot_names)

    # 3. Immunity
    init_immun = pd.read_hdf(lp.root / "data/init_immunity_0.5coverage_january.h5", key="immunity")
    init_immun = init_immun.set_index("dot_name").loc[dot_names]
    init_immun = init_immun[init_immun["period"] == start_year]

    # 4. Initial infection seeding
    init_prevs = np.zeros(len(dot_names))
    prev_indices = [i for i, dot_name in enumerate(dot_names) if init_region in dot_name]
    if not prev_indices:
        raise ValueError(f"No nodes found containing '{init_region}'")
    init_prevs[prev_indices] = init_prev
    print(f"[INFO] Seeding infection in {len(prev_indices)} nodes at {init_prev:.3f} prevalence.")

    # 5. SIA schedule
    start_date = lp.date(f"{start_year}-01-01")
    historic = pd.read_csv(lp.root / "data/sia_historic_schedule.csv")
    future = pd.read_csv(lp.root / "data/sia_scenario_1.csv")
    sia_schedule = lp.process_sia_schedule_polio(pd.concat([historic, future]), dot_names, start_date)

    # 6. Demographics and risk
    df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
    df_comp = df_comp[df_comp["year"] == start_year]
    pop = df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values * pop_scale
    cbr = df_comp.set_index("dot_name").loc[dot_names, "cbr"].values
    ri = df_comp.set_index("dot_name").loc[dot_names, "ri_eff"].values
    sia_re = df_comp.set_index("dot_name").loc[dot_names, "sia_random_effect"].values
    reff_re = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values
    sia_prob = lp.calc_sia_prob_from_rand_eff(sia_re)
    r0_scalars = lp.calc_r0_scalars_from_rand_eff(reff_re)

    # 7. Validate all arrays match
    assert all(len(arr) == len(dot_names) for arr in [dist_matrix, init_immun, centroids, init_prevs, pop, cbr, ri, sia_prob, r0_scalars])

    # 8. Base parameters (can be overridden)
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
            "centroids": centroids,
            "vx_prob_ri": ri,
            "sia_schedule": sia_schedule,
            "vx_prob_sia": sia_prob,
        }
    )

    # 9. Inject Optuna trial params if any exist
    if Path("params.json").exists():
        with open("params.json") as f:
            optuna_params = json.load(f)
        print("[INFO] Loaded Optuna trial params:", optuna_params)
        pars += optuna_params

    # 10. Run sim
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM, lp.RI_ABM, lp.SIA_ABM]

    print("[INFO] Running simulation...")
    sim.run()

    if save_plots:
        sim.plot(save=True, results_path=results_path)
    if save_data:
        Path(results_path).mkdir(parents=True, exist_ok=True)
        lp.save_results_to_csv(sim, filename=results_path + "/simulation_results.csv")

    sc.printcyan("Done.")


# ---------------------------- CLI ENTRY ----------------------------
if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True, help="Path to base model config YAML")
    parser.add_argument("--params-file", type=str, default="params.json", help="Trial parameter JSON file")
    parser.add_argument("--results-path", type=str, default="simulation_results.csv", help="Path to simulation results")
    args = parser.parse_args()

    # Load base config
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    # Load suggested parameter overrides (optional)
    params = {}
    if Path(args.params_file).exists():
        with open(args.params_file) as f:
            params = json.load(f)

    # Merge with precedence to Optuna params
    config = {**model_config, **params}
    if args.results_path:
        config["results_path"] = args.results_path
    setup_sim(config=config)
