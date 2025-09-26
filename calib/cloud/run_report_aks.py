import subprocess
import sys
import time
from pathlib import Path

import cloud_calib_config as cfg
import optuna

sys.path.append(str(Path(__file__).resolve().parent.parent))
from report import plot_likelihood_contribution_best
from report import plot_likelihood_contribution_by_param
from report import plot_likelihoods_vs_params
from report import plot_likelihoood_slices
from report import plot_mutual_information
from report import plot_optimization_history
from report import plot_quadratic_fit
from report import plot_runtimes
from report import plot_targets
from report import save_study_results


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3307:3306"])
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

        print("📊 Plotting target comparisons (actual vs. predicted) for top trial(s)...")
        plot_targets(study, n=10, output_dir=results_path)

        print("📈 Plotting optimization history...")
        plot_optimization_history(study, output_dir=results_path)

        print("📊 Plotting runtimes...")
        plot_runtimes(study, output_dir=results_path)

        print("📊 Plotting likelihood contribution for best trial...")
        plot_likelihood_contribution_best(study, output_dir=Path(results_path), use_log=True)

        print("📊 Plotting likelihood contribution by parameter...")
        plot_likelihood_contribution_by_param(study, output_dir=results_path)

        print("📊 Plotting likelihood slices...")
        plot_likelihoood_slices(study, output_dir=results_path)

        print("📊 Plotting likelihoods vs params...")
        plot_likelihoods_vs_params(study, output_dir=Path(results_path), use_log=True)

        print("📊 Plotting quadratic fit quality...")
        plot_quadratic_fit(study, output_dir=results_path)

        print("📊 Plotting mutual information analysis...")
        plot_mutual_information(study, output_dir=results_path)

        # print("📊 Running top trials on COMPS...")
        # from report import run_top_n_on_comps
        # from report import sweep_seed_best_comps
        # run_top_n_on_comps(study, n=1, output_dir=results_path)
        # sweep_seed_best_comps(study, output_dir=results_path)

    finally:
        print("🧹 Cleaning up port forwarding...")
        pf_process.terminate()
        print("🎉 Done!")


if __name__ == "__main__":
    main()
