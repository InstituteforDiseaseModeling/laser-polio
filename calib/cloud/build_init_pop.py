from pathlib import Path

import yaml

import laser_polio as lp

# Path to your original config
input_config_path = Path("calib/model_configs/config_nigeria_6y_init_pop_missed_pop.yaml")
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

# Path to save the new init_pop file
init_pop_output_path = output_dir / "init_pop_nigeria_6y.h5"

# Load and modify config
with open(input_config_path) as f:
    config = yaml.safe_load(f)

# Remove the init_pop_file to force fresh population creation
config["init_pop_file"] = None

# Set seed
config["seed"] = 117  # Set a specific seed for reproducibility

# Ensure output path is defined
config["results_path"] = str(output_dir)

# Run sim and save init_pop
print("🔄 Building new init_pop file...")
sim = lp.run_sim(config=config, save_init_pop=True, run=False, verbose=1)

# Rename file to standard name, safely
default_path = output_dir / "init_pop.h5"
if default_path.exists():
    if init_pop_output_path.exists():
        init_pop_output_path.unlink()  # Delete if already exists
    default_path.rename(init_pop_output_path)
    print(f"✅ init_pop saved to: {init_pop_output_path}")
else:
    raise FileNotFoundError("Expected init_pop.h5 was not generated.")

# Delete the actual_data.csv if it exists, as we are not using it here
actual_data_file = output_dir / "actual_data.csv"
if actual_data_file.exists():
    actual_data_file.unlink()
    print("🗑️ Deleted actual_data.csv as it is not needed for this operation.")
