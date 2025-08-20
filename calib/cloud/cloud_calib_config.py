# calib_job_config.py
from pathlib import Path

import yaml

# ------------------- USER CONFIGS -------------------

# # Goal: Start with 4 pars
# job_name = "lpsk5"
# study_name = "calib_nigeria_7y_2017_underwt_region_groupings_20250814"
# model_config = "config_nigeria_7y_2017_region_groupings.yaml"
# calib_config = "r0_k_ssn_wts.yaml"

# # Goal: switch to radiation_k_log10 & add max_migr_frac
# job_name = "lpsk6"
# study_name = "calib_nigeria_7y_2017_underwt_regions_maxmigrfrac_20250818"
# model_config = "config_nigeria_7y_2017_regions_sansmaxmigrfrac.yaml"
# calib_config = "r0_k_ssn_wts_maxmigrfrac.yaml"

# Goal: switch to radiation_k_log10 & add max_migr_frac, but set seasonal_peak_doy based on last calib
job_name = "lpsk8"
study_name = "calib_nigeria_7y_2017_underwt_regions_maxmigrfrac_setssnpeakdoy_20250818_forreal"
model_config = "config_nigeria_7y_2017_regions_sansmaxmigrfrac_setssnpeakdoy.yaml"
calib_config = "r0_k_ssnamp_wts_maxmigrfrac.yaml"

fit_function = "log_likelihood"
n_trials = 1  # Number of trials to run per pod
n_replicates = 1  # Number of replicates to run for each trial
parallelism = 200  # The number of pods (i.e., jobs) to run in parallel
completions = 5000  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"

# ---------------------------------------------------

# Default settings
namespace = "default"
image = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"

# Define the path to the YAML file with the storage URL from the docs
storage_path = Path("calib/cloud/local_storage.yaml")

# Try loading the storage URL from YAML, fallback to env var
storage_url = None
if storage_path.exists():
    storage = yaml.safe_load(storage_path.read_text())
    storage_url = storage.get("storage_url")
# Safety check
print(f"Storage URL: {storage_url}")
if storage_url is None:
    raise RuntimeError("Missing STORAGE_URL in local_storage.yaml")
