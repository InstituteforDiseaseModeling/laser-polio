import sys
import json
from pathlib import Path

from calib.cloud import cloud_calib_config as cfg
import sciris as sc

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
platform = Platform("COMPS", endpoint="https://comps.idmod.org", environment="CALCULON", type="COMPS")

# Build base command (run 1 trial per COMPS simulation)
command = CommandLine(
    f"singularity exec --env POLIO_ROOT='Assets' --no-mount /app Assets/laser-polio_latest.sif "
    f"python3 Assets/calib/calibrate.py "
    f"--study-name {cfg.study_name} "
    f"--n-trials 1 "
    #f"--model-config Assets/calib/model_configs/{cfg.model_config} "
    f"--model-config {cfg.model_config} "
    #f"--calib-config Assets/calib/calib_configs/{cfg.calib_config} "
    f"--calib-config {cfg.calib_config} "
    f"--fit-function {cfg.fit_function} "
    "--results-path '.' "
    "--actual-data-file /app/data/compiled_cbr_pop_ri_sia_underwt_africa.csv"
)

print( str( command ) )

# Create task
calib_task = CommandTask(command=command)
calib_task.common_assets.add_assets(AssetCollection.from_id_file("calib/comps/laser.id"))
calib_task.common_assets.add_assets(AssetCollection.from_directory("calib", relative_path="calib"))
calib_task.common_assets.add_assets(AssetCollection.from_directory("data", relative_path="data"))

# Optional: mount shared PVC if used
# calib_task.common_assets.add_directory("/shared")

# Build simulations (one per calibration worker)
simulations = []
for i in range(cfg.completions):
    sim = Simulation(task=calib_task)
    sim.name = f"calib-worker-{i}"
    sim.tags.update({
        "job": cfg.job_name,
        "study": cfg.study_name,
        "worker": i
    })
    add_schedule_config(
        sim,
        command=command,
        NumNodes=1,
        NumCores=12,
        NodeGroupName="idm_abcd",
        Environment={
            "NUMBA_NUM_THREADS": "12",
            "PYTHONPATH": "$PYTHONPATH:$PWD/Assets",
            "STORAGE_URL": "mysql://optuna:securepassword@10.24.14.21:3306/optunaDatabase",
            #"POLIO_ROOT": "."
        }
    )
    simulations.append(sim)

# Create experiment
experiment = Experiment(
    name=f"laser-calib-comps-{cfg.study_name}",
    simulations=simulations,
    tags={"source": "optuna", "mode": "comps-calibration"}
)

# Submit to COMPS
experiment.run(wait_until_done=False, priority="AboveNormal")

# Save experiment ID
output_path = Path("results") / cfg.study_name
output_path.mkdir(parents=True, exist_ok=True)
exp_id_path = output_path / "comps_exp.id"
experiment.to_id_file(exp_id_path)

sc.printgreen(f"✅ Submitted {len(simulations)} calibration workers to COMPS for study '{cfg.study_name}'.")
