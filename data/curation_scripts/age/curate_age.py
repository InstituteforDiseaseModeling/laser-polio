import geopandas as gpd
import pandas as pd

pd.set_option("display.max_columns", None)

### Import data
age_path = "data/curation_scripts/age/WPP2024_POP_F02_1_POPULATION_5-YEAR_AGE_GROUPS_BOTH_SEXES.xlsx"
age = pd.read_excel(age_path, sheet_name="Estimates", skiprows=16, skipfooter=16)
age.rename(columns={"ISO3 Alpha-code": "iso_3_code"}, inplace=True)

### Curate the spatial data
# Add in who_region column based on matches to the shapes file
shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")
shp = shp[["who_region", "adm0_name", "iso_3_code"]]
# Filter the age DataFrame based on the Location column having a match in the shp adm0_name column
age = age[age["iso_3_code"].isin(shp["iso_3_code"])]
# Merge the age DataFrame with the shp GeoDataFrame on the adm0_name column
age = age.merge(shp[["iso_3_code", "adm0_name"]], on="iso_3_code", how="left")

### Curate the year data
# Filter the years based on years included in the pop file
pop = pd.read_csv("data/curation_scripts/pop/dpt_district_summaries_curated.csv")
# Get the range of years in the pop DataFrame
year_min = pop["year"].min()
year_max = pop["year"].max()
# Filter the age DataFrame to include only the years within the range of years in the pop DataFrame
age = age[age["Year"].between(year_min, year_max)]

### Cleanup
# Select the ISO3 Alpha-code, Year, and the age groups columns
age = age[
    [
        "adm0_name",
        "iso_3_code",
        "Year",
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80-84",
        "85-89",
        "90-94",
        "95-99",
        "100+",
    ]
]
# Stack the age groups columns
age = age.melt(id_vars=["adm0_name", "iso_3_code", "Year"], var_name="age_group", value_name="population")
age = age.reset_index(drop=True)
print(age.head())

### Export the curated data
age.to_csv("data/age_africa.csv", index=False)

print("Done.")
