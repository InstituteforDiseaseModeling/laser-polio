# Calibration Framework for Laser-Polio Model

This directory contains a modular Optuna-based pipeline for calibrating polio simulations using the `laser_polio` agent-based model. It is designed to be flexible, testable, and extensible for different scoring functions, summary metrics, and model configurations.

---

## 🗂 File Structure and Module Overview

| File / Module     | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `calibrate.py`    | Main CLI entrypoint to run a full calibration loop locally              |
| `worker.py`       | Sets up and runs Optuna trials; loads config, study, scoring, and targets |
| `objective.py`    | Defines the Optuna `objective()` function that runs the model and scores output |
| `targets.py`      | Contains `calc_calib_targets*()` functions for summarizing model output |
| `scoring.py`      | Contains `compute_fit()` and `compute_log_likelihood_fit()` methods      |
| `report.py`       | Saves Optuna results and generates HTML plots using Plotly               |
| `calib_db.py`     | Configures Optuna storage (SQLite or MySQL)                              |

---

## How to Run Calibration Locally

### Step 1: Prepare Config Files
Make sure you have:
- A **calibration config** YAML (e.g., `calib/calib_configs/r0_k_seasonality.yaml`)
- A **model config** YAML (e.g., `calib/model_configs/config_nigeria_popscale0.01.yaml`)

### Step 2: Run calibrate.py

## How to run locally in a Docker container
???

## How to run remotely on a cluster
???

To build docker image, run docker build command from main directory.


# To calibrate on AKS...
1. You'll need to create a file called calib/cloud/local_storage.yaml that specifies the storage_url of the Optuna database. For security reasons, this file has been added to .gitignore.
2. The formatting is as follow:
`storage_url: "STORAGE_URL"`
3. See the docs for instructions on how to get the storage url.
