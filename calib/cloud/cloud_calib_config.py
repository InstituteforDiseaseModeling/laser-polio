# calib_job_config.py
from pathlib import Path

import yaml

# ------------------- USER CONFIGS -------------------

# job_name = "laser-polio-sk1"
# study_name = "calib_nigeria_6y_2018_underwt_grav_ipv_dirichlet_20250618"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv.yaml"
# calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet.yaml"

# job_name = "laser-polio-sk2"
# study_name = "calib_nigeria_6y_2018_underwt_grav_ipv_kanoboost_dirichlet_20250618"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_kanoboost.yaml"
# calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet.yaml"

# job_name = "laser-polio-sk3"
# study_name = "calib_nigeria_6y_2018_pim_grav_ipv_dirichlet_20250618"
# model_config = "config_nigeria_6y_2018_pim_gravity_zinb_ipv.yaml"
# calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet.yaml"

# job_name = "laser-polio-sk4"
# study_name = "calib_nigeria_6y_2018_pim_grav_ipv_dirichlet_3periods_earlierandmoreseeds_20250619"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds.yaml"
# calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet.yaml"

# job_name = "laser-polio-sk5"
# study_name = "calib_nigeria_6y_2018_pim_grav_ipv_dirichlet_3periods_earlierandmoreseeds_immunadj_20250619"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_immunadj.yaml"
# calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet.yaml"

# job_name = "laser-polio-sk6"
# study_name = "calib_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_alt_wts_20250701"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_alt.yaml"
# calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet_wts.yaml"

# job_name = "laser-polio-sk7"
# study_name = "calib_nigeria_6y_2018_underwt_gravity_zinb_ipv_wts_vxtranszero_20250703"
# model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_alt_zerovxtrans.yaml"
# calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet_wts.yaml"

job_name = "laser-polio-sk8"
study_name = "calib_nigeria_6y_2018_underwt_gravity_zinb_ipv_wts_vxtrans_20250703"
model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_alt.yaml"
calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet_wts.yaml"

fit_function = "log_likelihood"
n_trials = 1  # Number of trials to run per pod
n_replicates = 1  # Number of replicates to run for each trial
parallelism = 100  # The number of pods (i.e., jobs) to run in parallel
completions = 2  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"

# ---------------------------------------------------

# Default settings
namespace = "default"
image = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"
storage_url="mysql+pymysql://optuna:superSecretPassword@localhost:3306/optunaDatabase"
