import laser_polio as lp

calib_label = "calib_nigeria_r0"  # Used as the label for the calibration study & the results folder
calib_pars = "configs/calib/calib_r0.py"  # Path to the calibration parameters
model = "configs/model/sim_nigeria.py"  # Path to the model script, or a fn so we can modify features like save_plots, etc.
objective = lp.process_data()  # Objective function, or path to it???
gof = lp.compute_fit(use_squared=True)  # Goodness of fit function, or path to it???
n_trials = 1000  # Number of trials for the calibration
keep_calib_db = True  # Whether to keep the calibration database

if __name__ == "__main__":
    # Some sorta function to process runtime pars like n_trials, keep_calib_db, calib_label, etc.
    # Secret sauce placeholder

    # Calibrate
    calibrator = lp.Calibrator(
        model=model,
        calib_pars=calib_pars,
        objective=objective,
        gof=gof,
        label=calib_label,
    )
    calibrator.calibrate()

    # Export results
    results_folder = lp.root / "results" / calib_label
    results_folder.mkdir(parents=True, exist_ok=True)
    lp.export_calib_results(calibrator, results_path=results_folder)
    lp.plot_calib_results(calibrator, results_path=results_folder)

    print("Calibration complete!")
