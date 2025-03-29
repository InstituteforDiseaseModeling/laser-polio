import geopandas as gpd
import pandas as pd

from laser_polio.utils import clean_strings

pd.set_option("display.max_columns", None)
# Load the SIA calendar
df = pd.read_csv("data/curation_scripts/sia/sia_district_rows.csv", index_col=0)

# Load the adm0 shape to get the WHO region names
shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")
shp_who = shp[["ADM0_NAME", "WHO_REGION"]]
# Merge the cbr DataFrame with the shp GeoDataFrame on the ISO3_code column
df = df.merge(shp_who, left_on="ADM0_NAME", right_on="ADM0_NAME", how="left")

# Filter to only the rows with a WHO_REGION (the shape is filtered to only the African countries)
df = df[df.WHO_REGION.notnull()]

# Generate dot-separated 'dot_name'
dot_name_cols = ["WHO_REGION", "ADM0_NAME", "ADM1_NAME", "ADM2_NAME"]
df[dot_name_cols] = df[dot_name_cols].map(clean_strings)
df["dot_name"] = df[dot_name_cols].agg(":".join, axis=1)
# Manually update dot_name for duplicates
df.loc[df["GUID"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "dot_name"] = "AFRO:GUINEA_BISSAU:BAFATA:BAMBADINCA2"

# Process dates
df = df.rename(columns={"start_date": "date"})

# Process the age groups
# Load the age_ranges dictionary
age_ranges = pd.read_csv("data/curation_scripts/sia/age_ranges.csv")
# Calculate the age group in days
age_ranges["agemin_d"] = (age_ranges["agemin_y"] * 365).astype(int)
age_ranges["agemax_d"] = (age_ranges["agemax_y"] * 365).astype(int)
# Merge in the year columns
df = df.merge(age_ranges[["agegroup", "agemin_d", "agemax_d"]], left_on="agegroup", right_on="agegroup", how="left")
# Rename the age columns
df = df.rename(columns={"agemin_d": "age_min", "agemax_d": "age_max"})

# Tidy up
# Select only the date, dot_name, age_min, and age_max columns
df = df[["date", "dot_name", "GUID", "age_min", "age_max", "vaccinetype"]]
# Reset the index
df = df.reset_index(drop=True)
print(df.head())

# Save the full curated dataframe
# Output:
# date,dot_name,age_min,age_max
# 2025-02-01,AFRO:NIGERIA:SOKOTO:BINJI,0,1825
# 2025-02-01,AFRO:NIGERIA:SOKOTO:BODINGA,0,1825
df.to_csv("data/sia_historic_schedule.csv", index=False)

print("Done.")
