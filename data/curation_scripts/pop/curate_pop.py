import pandas as pd

pd.set_option("display.max_columns", None)

"""
This script is used to curate the population data from the polio-immunity-mapping project.
See the README.md file for more information on how to generate the pop files.
"""

# Load the raw data
pop = pd.read_csv("data/curation_scripts/pop/demog_data_POLIS_ADM02_yearly_cbr.csv")
# Curated by Kurt
# Uses the same shapefile as us (shp_africa_low_res.gpkg) so no need to curate admin names

# pop.columns = pop.columns.str.lower()

# # Clean the specified columns
# columns_to_clean = ["adm0_name", "adm1_name", "adm2_name"]
# pop[columns_to_clean] = pop[columns_to_clean].map(clean_strings)

# # Add in who_region column based on matches to the shapes file
# shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")

# # Filter the pop DataFrame based on the adm0_name column having a match in the shp adm0_name column
# pop = pop[pop["adm0_name"].isin(shp["adm0_name"])]

# # Merge the pop DataFrame with the shp GeoDataFrame on the adm0_name column
# pop = pop.merge(shp[["adm0_name", "who_region"]], on="adm0_name", how="left")

# # Generate dot-separated 'dot_name'
# dot_name_cols = ["who_region", "adm0_name", "adm1_name", "adm2_name"]
# pop["dot_name"] = pop[dot_name_cols].agg(":".join, axis=1)
# print(pop.head())

# # Check for dot_name and year combination duplicates
# if len(pop) != len(pop[["dot_name", "year"]].drop_duplicates()):
#     print("Before manual curation, the dot_name and year combinations are not unique.")
#     # Print the rows with duplicates
#     duplicates = pop[pop.duplicated(subset=["dot_name", "year"], keep=False)]
#     print(duplicates)

# # Manually update dot_name for duplicates
# pop.loc[pop["guid"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "dot_name"] = "AFRO:GUINEA_BISSAU:BAFATA:BAMBADINCA2"
# pop.loc[pop["guid"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "adm2_name"] = "BAMBADINCA2"

# Check again for dot_name and year combination duplicates
if len(pop) != len(pop[["dot_name", "year"]].drop_duplicates()):
    print("After manual curation, the dot_name and year combinations are not unique.")
    # Print the rows with duplicates
    duplicates = pop[pop.duplicated(subset=["dot_name", "year"], keep=False)]
    print(duplicates)
else:
    print("After manual curation, the dot_name and year combinations are unique.")

# Test that all values in pop.who_region are not NaN or NA
# assert not pop["who_region"].isna().any(), "There are NaN values in the who_region column"
assert not pop["dot_name"].isna().any(), "There are NaN values in the dot_name column"

# # Save the full curated dataframe
# pop.to_csv("data/curation_scripts/pop/dpt_district_summaries_curated.csv", index=False)

# # Filter to only the columns we need & Save the cleaned dataframe
# pop = pop[['dot_name', 'year', 'under5_pop']]
# pop.to_csv('data/pop_africa_u5.csv', index=False)
print("Done")
