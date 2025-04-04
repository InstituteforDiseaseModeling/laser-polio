import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rastertools
from PIL import Image
from rastertools import raster_clip

print(f"rastertools version v{rastertools.__version__}")

# Increase the MAX_IMAGE_PIXELS limit to handle large images
Image.MAX_IMAGE_PIXELS = None

# Define inputs
raster_path = "data/curation_scripts/individual_risk/IHME_CGF_UNDERWEIGHT_2019_MEAN.tif"
shp_path = "data/shp_africa_low_res.gpkg"
output_file = "data/curation_scripts/individual_risk/underwt_u5_2019.csv"

# Check if shp file exists
if not os.path.exists(shp_path):
    print(f"Shapefile not found at {shp_path}. Loading geojson file and saving as shapefile.")
    shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")
    # Save as a shapefile
    shp.to_file(shp_path)


# Define the mean summary function
def mean_summary_func(v: np.ndarray) -> float:
    """Calculate the mean of an array."""
    return float(np.mean(v))


# Clip the raster with the shapefile for each dot_name
result = raster_clip(
    raster_file=raster_path, shape_stem=shp_path, shape_attr="dot_name", summary_func=mean_summary_func, include_latlon=True, quiet=True
)

### Curate the df
df = pd.DataFrame.from_dict(result, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "dot_name", "pop": "prop_underwt"}, inplace=True)
df["prop_underwt"] = df["prop_underwt"] / 100

###  Validate the output

# Check that the shp and df are the same len
shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")
assert len(shp) == len(df), "The number of rows in the output does not match the number of shapes in the shapefile."

# Fill in missing values with the mean from the next higher admin level
# Merge the admin names from shp to the df
df = df.merge(shp[["dot_name", "adm0_name", "adm1_name", "adm2_name"]], on="dot_name", how="left")
# Count number of NA values
n_na = df["prop_underwt"].isna().sum()
if n_na > 0:
    print(f"There are {n_na} missing values in the prop_underwt column. Filling in missing values with the mean from higher admin levels.")
    print("Rows with missing values:")
    # Print the rows with missing values
    print(df[df.isna().any(axis=1)])
# Fill in missing values with the mean from higher admin levels
df.loc[:, "prop_underwt"] = df.groupby(["adm0_name", "adm1_name"])["prop_underwt"].transform(lambda x: x.fillna(x.mean()))
df.loc[:, "prop_underwt"] = df.groupby(["adm0_name"])["prop_underwt"].transform(lambda x: x.fillna(x.mean()))
# Check for missing values again
assert df["prop_underwt"].isna().sum() == 0, (
    "There are still missing values in the immunity_ri_nOPV2 column after taking the adm1 or adm0 mean."
)

# Export the file
df.to_csv(output_file, index=False)

print("Done.")
