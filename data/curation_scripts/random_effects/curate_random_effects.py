import geopandas as gpd
import pandas as pd

from laser_polio.utils import clean_strings

# Load random effects for R_eff & SIA coverage from the regression model
# Source: Kurt_sharing\step04b_sia_randeffect\20241220_regression
df = pd.read_csv("data/curation_scripts/random_effects/random_effects_summary_2024-12-20.csv")

# Clean the specified columns
columns_to_clean = ["adm0_name", "adm1_name"]
df[columns_to_clean] = df[columns_to_clean].map(clean_strings)

# Load the shape data
shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm1")

# Filter to only the rows in the df that match the adm0_name in the shapefile
df = df[df["adm0_name"].isin(shp["adm0_name"])]

# Ensure all the admin names are in in the shapefile and vice versa
df["adm0_name"].isin(shp["adm0_name"]).all()
shp["adm0_name"].isin(df["adm0_name"]).all()

df["adm1_name"].isin(shp["adm1_name"]).all()
shp["adm1_name"].isin(df["adm1_name"]).all()

missing_df_adm0_names = df[~df["adm0_name"].isin(shp["adm0_name"])]["adm0_name"].unique()
missing_df_adm1_names = df[~df["adm1_name"].isin(shp["adm1_name"])]["adm1_name"].unique()

# df[df["adm1_name"] == "RUMONGE"]
# shp1 = gpd.read_file("data/curation_scripts/shp/shape1.gpkg")
# shp1[shp1["adm1_name"] == "RUMONGE"]

# Filter to only the columns we need & Save the cleaned dataframe
df = df[["adm0_name", "adm1_name", "r_effect_mean", "sia_effect_mean"]]
df = df.rename(
    columns={
        "adm0_name": "adm0_name",
        "adm1_name": "adm1_name",
        "r_effect_mean": "reff_random_effect",
        "sia_effect_mean": "sia_random_effect",
    }
)
print(df.head())

# Save the full curated dataframe
df.to_csv("data/curation_scripts/random_effects/random_effects_curated.csv", index=False)


print("Done.")
