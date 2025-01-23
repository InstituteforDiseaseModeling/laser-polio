import importlib
from pathlib import Path

# Use Path for directory and file paths
TEST_DIR = Path(__file__).parent  # Get the directory of the test script
# TEST_DATA_DIR = TEST_DIR / "test_data"  # Adjust path
TEST_DATA_DIR = TEST_DIR

psi_data = TEST_DATA_DIR / "pred_psi_processed.csv"
# psi_data = "psi_synth_allsame.csv"
# age_data = pkg_resources.path('idmlaser_cholera', 'USA-pyramid-2023.csv')  # meh
age_data = str(TEST_DATA_DIR / "nigeria_pyramid.csv")
seasonal_dynamics = TEST_DATA_DIR / "pred_seasonal_dynamics_processed.csv"
immune_decay_params = TEST_DATA_DIR / "immune_decay_params.json"
laser_cache = TEST_DATA_DIR / "laser_cache"
# wash_theta = "param_theta_WASH.csv"
# population_file = "nigeria.py"
# population_file = "nigeria_onenode.py"
# population_file = "synth_10_allsame.py"
# population_file = "synth_25.py"
population_file = TEST_DATA_DIR / "synth_small_ssa.py"


def load_population_data():
    if population_file is None:
        raise ValueError("A data file path must be specified.")

    # Check if the data file is a Python module
    if population_file.is_file() and population_file.suffix == ".py":
        spec = importlib.util.spec_from_file_location("location_data", str(population_file))
        location_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(location_data)

        return location_data.run()
    else:
        raise ValueError(f"Invalid data file '{population_file!s}'. It must be a Python file.")
