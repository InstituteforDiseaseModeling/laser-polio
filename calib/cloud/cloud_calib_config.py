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

#job_name = "laser-polio-sk5"
#study_name = "calib_nigeria_6y_2018_pim_grav_ipv_dirichlet_3periods_earlierandmoreseeds_immunadj_20250619"
#model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_immunadj.yaml"
#calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet.yaml"

job_name = "laser-polio-jb-june-25"
study_name = "calib_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_alt_wts_20250623"
model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_alt.yaml"
calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet_wts.yaml"

fit_function = "log_likelihood"
n_trials = 1  # Number of trials to run per pod
n_replicates = 1  # Number of replicates to run for each trial
parallelism = 50  # The number of pods (i.e., jobs) to run in parallel
#completions = 20000  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"
completions = 2  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"
# ---------------------------------------------------
storage_url = "mysql+pymysql://optuna:superSecretPassword@127.0.0.1:3306/optunaDatabase"

