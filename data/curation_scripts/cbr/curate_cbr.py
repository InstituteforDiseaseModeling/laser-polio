import geopandas as gpd
import pandas as pd

# Load the WPP data with CBRs
cbr = pd.read_csv(
    "data/curation_scripts/cbr/WPP2024_Demographic_Indicators_Medium.csv.gz"
)  # From: https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format
cbr = cbr.rename(columns={"ISO3_code": "iso_3_code", "Time": "year", "CBR": "cbr"})
cbr = cbr[["Location", "iso_3_code", "year", "cbr"]]
print(cbr.head())

# Load the adm0 shapes file
shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")

# Merge the cbr DataFrame with the shp GeoDataFrame on the ISO3_code column
merged = shp.merge(cbr, left_on="iso_3_code", right_on="iso_3_code", how="left")

# Load the curated dpt dataset and use that year range to filter the cbr dataset
dpt = pd.read_csv("data/curation_scripts/pop/dpt_district_summaries_curated.csv")

# Filter merged to years found in the year column in dpt
merged = merged[merged["year"].isin(dpt["year"])]

# Filter to columns of interest
merged = merged[["adm0_name", "year", "cbr"]]
print(merged.head())

# Save the dataset
merged.to_csv("data/curation_scripts/cbr/cbr_curated.csv", index=False)

print("Done.")
