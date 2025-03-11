import geopandas as gpd
import pandas as pd

from laser_polio.utils import clean_strings

pd.set_option("display.max_columns", None)

"""
This script is used to curate the population data from the polio-immunity-mapping project.
See the README.md file for more information on how to generate the pop files.
"""

# Load the raw data
pop = pd.read_csv("data/curation_scripts/pop/dpt_district_summaries.csv")

# Clean the specified columns
columns_to_clean = ["ADM0_NAME", "ADM1_NAME", "ADM2_NAME"]
pop[columns_to_clean] = pop[columns_to_clean].map(clean_strings)

# Add in WHO_REGION column based on matches to the shapes file
shp = gpd.read_file("data/shp_africa_adm0.geojson")

# Filter the pop DataFrame based on the ADM0_NAME column having a match in the shp ADM0_NAME column
pop = pop[pop["ADM0_NAME"].isin(shp["ADM0_NAME"])]

# Merge the pop DataFrame with the shp GeoDataFrame on the ADM0_NAME column
pop = pop.merge(shp[["ADM0_NAME", "WHO_REGION"]], on="ADM0_NAME", how="left")

# Generate dot-separated 'dot_name'
dot_name_cols = ["WHO_REGION", "ADM0_NAME", "ADM1_NAME", "ADM2_NAME"]
pop["dot_name"] = pop[dot_name_cols].agg(":".join, axis=1)
print(pop.head())

# Check for dot_name and year combination duplicates
if len(pop) != len(pop[["dot_name", "year"]].drop_duplicates()):
    print("The dot_name and year combinations are not unique.")
    # Print the rows with duplicates
    duplicates = pop[pop.duplicated(subset=["dot_name", "year"], keep=False)]
    print(duplicates)

# Manually update dot_name for duplicates
pop.loc[pop["GUID"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "dot_name"] = "AFRO:GUINEA_BISSAU:BAFATA:BAMBADINCA2"
pop.loc[pop["GUID"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "ADM2_NAME"] = "BAMBADINCA2"

# Check again for dot_name and year combination duplicates
if len(pop) != len(pop[["dot_name", "year"]].drop_duplicates()):
    print("The dot_name and year combinations are not unique.")
    # Print the rows with duplicates
    duplicates = pop[pop.duplicated(subset=["dot_name", "year"], keep=False)]
    print(duplicates)

# Test that all values in pop.WHO_REGION are not NaN or NA
assert not pop["WHO_REGION"].isna().any(), "There are NaN values in the WHO_REGION column"
assert not pop["dot_name"].isna().any(), "There are NaN values in the dot_name column"

# Save the full curated dataframe
pop.to_csv("data/curation_scripts/pop/dpt_district_summaries_curated.csv", index=False)

# # Filter to only the columns we need & Save the cleaned dataframe
# pop = pop[['dot_name', 'year', 'under5_pop']]
# pop.to_csv('data/pop_africa_u5.csv', index=False)
print("Done")
