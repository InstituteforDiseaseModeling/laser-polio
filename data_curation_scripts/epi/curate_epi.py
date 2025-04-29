import geopandas as gpd
import pandas as pd

import laser_polio as lp

# Load data
df = pd.read_hdf("data/curation_scripts/epi/tsir_data_subset_type2_2025-04-21.h5", key="tsir_data")
shp = gpd.read_file(filename="data/shp_africa_low_res.gpkg", layer="adm2")

# Drop the administrative level columns from the df
df = df.drop(columns=["adm0_name", "adm1_name", "adm2_name"])
# Merge the shp & df to get the dot_name column
df = df.merge(shp[["guid", "dot_name"]], on="guid", how="left")
# Filter out the rows with missing dot_names
df = df[df["dot_name"].notnull()]
# Check for number of unique dot_names
assert len(shp) == df["dot_name"].nunique()
# Clean up & reorder the columns
df = df[["dot_name", "guid", "month_start", "cases", "es_samples", "es_positives"]]
# Drop any rows with month_start > today
df = df[lp.date(df["month_start"]) <= pd.Timestamp.now().date()]


# Save in Pandas-native HDF5 format
df.to_hdf("data/epi_africa_20250421.h5", key="epi", mode="w", format="table", complevel=5)

print("Done")
