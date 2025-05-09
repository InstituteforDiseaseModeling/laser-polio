import os
import sys
from pathlib import Path

import laser_polio as lp

if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))
sys.path.append(str(lp.root))
from calib.targets import calc_targets_temporal_regional_nodes

data_path = lp.root / "results" / "demo_nigeria_best_calib"

actual_data_path = data_path / "actual_data.csv"
sim_data_path = data_path / "simulation_results.csv"

actual = calc_targets_temporal_regional_nodes(
    filename=actual_data_path,
    model_config_path=None,
    is_actual_data=True,
)
predicted = calc_targets_temporal_regional_nodes(
    filename=sim_data_path,
    model_config_path=None,
    is_actual_data=False,
)

print("Done.")
