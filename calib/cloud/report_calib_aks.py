import sys
from pathlib import Path

import cloud_calib_config as cfg
import optuna

sys.path.append(str(Path(__file__).resolve().parent.parent))
from report import plot_stuff
from report import save_study_results

# STORAGE_URL2 = "mysql+pymysql://root@127.0.0.1:3306/optuna_db"
IMAGE_NAME = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"
# study_name = "calib_demo_nigeria2"
study = optuna.load_study(study_name=cfg.study_name, storage=cfg.storage_url)
study.storage_url = cfg.storage_url
study.study_name = cfg.study_name
save_study_results(study, Path(cfg.study_name))
plot_stuff(cfg.study_name, study.storage_url)
