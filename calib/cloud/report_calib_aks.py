import subprocess
import sys
import time
from pathlib import Path

import cloud_calib_config as cfg
import optuna

sys.path.append(str(Path(__file__).resolve().parent.parent))
from report import plot_stuff
from report import plot_targets  # or wherever it's defined
from report import save_study_results


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3306:3306"])
    time.sleep(3)  # wait for port-forward to take effect
    return pf


def main():
    pf_process = port_forward()
    try:
        print(f"📊 Loading study '{cfg.study_name}'...")
        study = optuna.load_study(study_name=cfg.study_name, storage=cfg.storage_url)
        study.storage_url = cfg.storage_url
        study.study_name = cfg.study_name

        results_path = Path("results") / cfg.study_name
        results_path.mkdir(parents=True, exist_ok=True)

        print("💾 Saving results...")
        save_study_results(study, output_dir=results_path)

        print("📈 Plotting results...")
        plot_stuff(cfg.study_name, study.storage_url, output_dir=results_path)

        print("📊 Plotting target comparisons...")
        plot_targets(study, output_dir=results_path)

    finally:
        print("🧹 Cleaning up port forwarding...")
        pf_process.terminate()


if __name__ == "__main__":
    main()
