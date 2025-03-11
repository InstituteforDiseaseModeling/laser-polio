import geopandas as gpd
import numpy as np
import pandas as pd

from laser_polio.utils import clean_strings

# Load the raw data
sia = pd.read_csv("data/curation_scripts/sia/relative_sia_impact_sia_random_effects.csv")

# Clean the specified columns
columns_to_clean = ["adm0_name", "adm1_name"]
sia[columns_to_clean] = sia[columns_to_clean].map(clean_strings)

# Add in WHO_REGION column based on matches to the shapes file
shp = gpd.read_file("data/shp_africa_adm0.geojson")

# Filter the sia DataFrame based on the ADM0_NAME column having a match in the shp ADM0_NAME column
sia = sia[sia["adm0_name"].isin(shp["ADM0_NAME"])]

# Filter to only the columns we need & Save the cleaned dataframe
sia = sia[["adm0_name", "adm1_name", "mean"]]
sia = sia.rename(columns={"adm0_name": "ADM0_NAME", "adm1_name": "ADM1_NAME", "mean": "sia_random_effect"})
print(sia.head())


# Transform the sia_random_effect column from logit to probability
def inverse_logit(x):
    return np.exp(x) / (1 + np.exp(x))


sia["sia_prob"] = inverse_logit(sia["sia_random_effect"])

# Save the full curated dataframe
sia.to_csv("data/curation_scripts/sia/sia_random_effects_curated.csv", index=False)

print("Done.")
