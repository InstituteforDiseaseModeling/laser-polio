# Load required libraries
library(reticulate)
library(dplyr)
library(sf)
# Use Python in R
pd <- import("pandas")

# Import the immunity estimates and save them as an h5 file
coverages <- c(0.5, 0.8)
for (coverage in coverages) {
  # Define the paths
  input_path <- sprintf("data/curation_scripts/init_immunity/immunity_age_groups_%.1fcoverage.rds", coverage)
  output_path <- sprintf("data/init_immunity_%.1fcoverage_january.h5", coverage)

  # Load the RDS file
  data <- readRDS(input_path) %>% as.data.frame()

  # Filter to serotype 2
  data <- data %>% filter(serotype == 'p2')

  # Filter to periods that are in January, e.g, period has no decimal
  data <- data %>% filter(period %% 1 == 0)

  # Filter the immunity file to guids that are in the shp
  shp <- st_read('data/shp_africa_low_res.gpkg', layer='adm2')
  data <- data %>% filter(guid %in% shp$guid)

  # Merge the dot_name column based on matching GUID column
  shp_names <- shp %>% select(guid, dot_name)
  shp_names <- shp_names %>% st_drop_geometry()
  data <- data %>% left_join(shp_names)
  data <- data %>% select(dot_name, everything())

  # Use Python's pandas to export in HDFStore format
  use_python("C:/github/laser-polio/.venv/Scripts/python.exe")  # Replace with the correct path to your Python executable

  # Pass the output_path variable to Python
  reticulate::py_run_string(sprintf("
import pandas as pd
import numpy as np

# Load data from R
r_data = r.data

# Convert to Pandas DataFrame
df = pd.DataFrame(r_data)

# Save in Pandas-native HDF5 format
df.to_hdf('%s', key='immunity', mode='w', format='table', complevel=9)
  ", output_path))
}

print("Done")
