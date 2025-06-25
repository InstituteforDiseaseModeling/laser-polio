import sys
import json
import copy
from pathlib import Path

from calib.cloud import cloud_calib_config as cfg

from idmtools.assets import Asset, AssetCollection
from idmtools.core.platform_factory import Platform
from idmtools.entities import CommandLine
from idmtools.entities.command_task import CommandTask
from idmtools.entities.experiment import Experiment
from idmtools.entities.simulation import Simulation
from idmtools_platform_comps.utils.scheduling import add_schedule_config

# Shared volume path (simulates SHARED_DIR from AKS)
SHARED_DIR = "/app/shared"  # or "/shared" depending on COMPS mount setup

# Connect to COMPS platform
platform = Platform("CALCULON", endpoint="https://comps.idmod.org", environment="CALCULON", type="COMPS")

# Build base command (run 1 trial per COMPS simulation)
command = CommandLine(
    f"singularity exec --no-mount /app Assets/laser-polio_latest.sif "
    f"python3 Assets/calibrate.py "
    f"--study-name {cfg.study_name} "
    f"--n-trials 1 "
    f"--model-config Assets/model_configs/{cfg.model_config} "
    f"--calib-config Assets/calib_configs/{cfg.calib_config} "
    f"--fit-function {cfg.fit_function}"
)

# Create task
calib_task = CommandTask(command=command)
calib_task.common_assets.add_assets(AssetCollection.from_id_file("calib/comps/laser.id"))
calib_task.common_assets.add_assets(AssetCollection.from_directory("calib"))

# Optional: mount shared PVC if used
# calib_task.common_assets.add_directory("/shared")

# Create experiment
experiment = Experiment(
    name=f"laser-calib-comps-{cfg.study_name}",
    tags={"source": "optuna", "mode": "comps-calibration"}
)
# Build simulations (one per calibration worker)
for i in range(cfg.completions):
    task = copy.deepcopy(calib_task)  # <- create a separate task per simulation 
    sim = Simulation(task=task)
    sim.name = f"calib-worker-{i}"
    sim.tags.update({
        "job": cfg.job_name,
        "study": cfg.study_name,
        "worker": i
    })
    experiment.simulations.append(sim)

    add_schedule_config(
        sim,
        command=command,
        NumNodes=1,
        NumCores=12,
        NodeGroupName="idm_a",
        Environment={
            "NUMBA_NUM_THREADS": "12",
            "PYTHONPATH": "$PYTHONPATH:$PWD/Assets", 
            "STORAGE_URL": "mysql://optuna:securepassword@10.24.14.21:3306/optunaDatabase" 
        }
    ) 

# Submit to COMPS
experiment.run(wait_until_done=False)

# Save experiment ID
output_path = Path("results") / cfg.study_name
output_path.mkdir(parents=True, exist_ok=True)
exp_id_path = output_path / "comps_exp.id"
experiment.to_id_file(exp_id_path)

print(f"✅ Submitted {len(experiment.simulations)} calibration workers to COMPS for study '{cfg.study_name}'.") 
