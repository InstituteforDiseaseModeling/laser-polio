import subprocess
import time
from pathlib import Path

import cloud_calib_config as cfg
import numpy as np
import optuna
import sciris as sc
import yaml

import laser_polio as lp


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3306:3306"])
    time.sleep(3)  # wait for port-forward to take effect
    return pf


def main():
    pf_process = port_forward()
    try:
        print(f"📊 Loading study '{cfg.study_name}'...")
        # Load the study
        study = optuna.load_study(study_name=cfg.study_name, storage=cfg.storage_url)
        best_params = study.best_trial.params

        with open(Path("calib/model_configs/") / cfg.model_config) as f:
            model_config = yaml.safe_load(f)
        model_config["results_path"] = "results/" + cfg.study_name
        model_config["save_plots"] = True
        model_config["plot_pars"] = True
        # pars = PropertySet(model_config)
        pars = sc.mergedicts(model_config, best_params)  # apply best trial overrides

        # Extract rand_seed from the best trial (which can have reps)
        rep_scores = study.best_trial.user_attrs["rep_scores"]
        best_idx = np.where(rep_scores == np.min(rep_scores))
        rand_seeds = study.best_trial.user_attrs["rand_seed"]
        rand_seed = rand_seeds[best_idx[0][0]]
        pars["seed"] = rand_seed

        # Run sim & save plots
        print("💫Running sim with best trial parameters...")
        lp.run_sim(pars, verbose=1)

    finally:
        print("🧹 Cleaning up port forwarding...")
        pf_process.terminate()
        print("🎉Done!")


if __name__ == "__main__":
    main()
