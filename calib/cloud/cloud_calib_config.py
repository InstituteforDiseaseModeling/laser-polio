# calib_job_config.py
from pathlib import Path

import yaml

# ------------------- USER CONFIGS -------------------

# # Goal: Try all core pars after adding cases_by_month
# job_name = "lpsk9"
# study_name = "calib_nigeria_7y_2017_r0_amp_doy_radk_mmf_nozi_underwt_20250916"
# model_config = "nigeria_7y_2017_regions_r0_radk_mmf_ssn_nozi_underwt.yaml"
# calib_config = "r0_amp_doy_radk_mmf.yaml"

# # Goal: Try all core pars after adding cases_by_month
# job_name = "lpsk10"
# study_name = "calib_nigeria_7y_2017_r0_amp_doy_radk_mmf_nozi_underwt_narrow_20250917"
# model_config = "nigeria_7y_2017_regions_r0_radk_mmf_ssn_nozi_underwt.yaml"
# calib_config = "r0_amp_doy_radk_mmf_narrow.yaml"

# Goal: Try calibrating core pars in West Africa
job_name = "lpsk10"
study_name = "calib_wa_7y_2017_r0_amp_doy_radk_mmf_nozi_underwt_20250924"
model_config = "wa_7y_2017_regions_r0_radk_mmf_ssn_nozi_underwt.yaml"
calib_config = "r0_amp_doy_radk_mmf.yaml"

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
