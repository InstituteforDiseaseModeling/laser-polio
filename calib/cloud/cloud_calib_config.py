# calib_job_config.py
from pathlib import Path

import yaml

# ------------------- USER CONFIGS -------------------

# job_name = "laser-polio-worker-sk"
# study_name = "calib_nigeria_3y_2018_pim_gravity_zinb_ipv_20250612"
# model_config = "config_nigeria_3y_2018_pim_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb.yaml"

# job_name = "laser-polio-worker-sk2"
# study_name = "calib_nigeria_4y_2020_pim_gravity_zinb_ipv_20250612"
# model_config = "config_nigeria_4y_2020_pim_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb.yaml"

# job_name = "laser-polio-worker-sk3"
# study_name = "calib_nigeria_3y_2018_underwt_gravity_zinb_ipv_20250612"
# model_config = "config_nigeria_3y_2018_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb_r0wtscalars.yaml"

# job_name = "laser-polio-worker-sk4"
# study_name = "calib_nigeria_4y_2020_underwt_gravity_zinb_ipv_20250612"
# model_config = "config_nigeria_4y_2020_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb_r0wtscalars.yaml"

# job_name = "laser-polio-worker-sk5"
# study_name = "calib_nigeria_6y_2018_underwt_gravity_zinb_ipv_20250613"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb_r0wtscalars.yaml"

# job_name = "laser-polio-worker-sk6"
# study_name = "calib_nigeria_7y_2017_underwt_gravity_zinb_ipv_20250613"
# model_config = "config_nigeria_7y_2017_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb_r0wtscalars.yaml"

# job_name = "laser-polio-worker-sk7"
# study_name = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_immunscalars_20250616"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb_r0wtscalars_immunscalars.yaml"

# job_name = "laser-polio-worker-sk8"
# study_name = "calib_nigeria_7y_2017_underwt_gravity_zinb_ipv_immunscalars_20250616"
# model_config = "config_nigeria_7y_2017_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb_r0wtscalars_immunscalars.yaml"

# job_name = "laser-polio-worker-sk9"
# study_name = "calib_nigeria_7y_2017_underwt_gravity_zinb_ipv_immunscalars_kanobounds_20250617"
# model_config = "config_nigeria_7y_2017_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_k_ssn_gravitykabc_zinb_r0wtscalars_immunscalars.yaml"

job_name = "laser-polio-sk10"
study_name = "calib_nigeria_6y_2018_underwt_grav_ipv_20250617"
model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv.yaml"
calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs.yaml"

fit_function = "log_likelihood"
n_trials = 1  # Number of trials to run per pod
n_replicates = 1  # Number of replicates to run for each trial
parallelism = 100  # The number of pods (i.e., jobs) to run in parallel
completions = 20000  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"

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
