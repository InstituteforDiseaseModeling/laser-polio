import subprocess
import sys
import time
from pathlib import Path

import cloud_calib_config as cfg
import optuna

sys.path.append(str(Path(__file__).resolve().parent.parent))
import yaml
from report import plot_likelihoods
from report import plot_trial_targets

# ------------------- USER CONFIGS -------------------

# Trial number to plot (can be overridden by command line argument)
trial_number = 575

# ---------------------------------------------------


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3307:3306"])
    time.sleep(3)  # wait for port-forward to take effect
    return pf


def main():
    global trial_number

    # Use command line argument if provided, otherwise use config default
    if len(sys.argv) > 1:
        if len(sys.argv) != 2:
            print("Usage: python run_report_aks_trial.py [trial_number]")
            print("Example: python run_report_aks_trial.py 42")
            print(f"Or edit the trial_number in the USER CONFIGS section (currently: {trial_number})")
            sys.exit(1)

        try:
            trial_number = int(sys.argv[1])
        except ValueError:
            print("Error: Trial number must be an integer")
            sys.exit(1)

    print(f"📊 Using trial number: {trial_number}")

    pf_process = port_forward()
    try:
        print(f"📊 Loading study '{cfg.study_name}'...")
        study = optuna.load_study(study_name=cfg.study_name, storage=cfg.storage_url)
        study.storage_url = cfg.storage_url
        study.study_name = cfg.study_name
        results_path = Path("results") / cfg.study_name / f"trial_{trial_number}"
        results_path.mkdir(parents=True, exist_ok=True)

        # Check if trial exists
        trial = None
        for t in study.trials:
            if t.number == trial_number:
                trial = t
                break

        if trial is None:
            print(f"❌ Trial {trial_number} not found in study")
            sys.exit(1)

        if trial.state != optuna.trial.TrialState.COMPLETE:
            print(f"❌ Trial {trial_number} is not completed (state: {trial.state})")
            sys.exit(1)

        print(f"✅ Found trial {trial_number} with value: {trial.value}")

        with open(Path("calib/model_configs/") / cfg.model_config) as f:
            model_config = yaml.safe_load(f)
            start_year = model_config["start_year"]

        print(f"📊 Plotting trial {trial_number} targets...")
        plot_trial_targets(study, trial_number=trial_number, output_dir=results_path, start_year=start_year, model_config=model_config)

        print(f"📊 Plotting trial {trial_number} likelihoods...")
        plot_likelihoods(study, trial_number=trial_number, output_dir=Path(results_path), use_log=True)

    finally:
        print("🧹 Cleaning up port forwarding...")
        pf_process.terminate()
        print("🎉 Done!")


if __name__ == "__main__":
    main()
