import geopandas as gpd
import pandas as pd

pd.set_option("display.max_columns", None)

### Import data
age_path = "data/curation_scripts/age/WPP2024_POP_F02_1_POPULATION_5-YEAR_AGE_GROUPS_BOTH_SEXES.xlsx"
age = pd.read_excel(age_path, sheet_name="Estimates", skiprows=16, skipfooter=16)
age.rename(columns={"ISO3 Alpha-code": "ISO_3_CODE"}, inplace=True)

### Curate the spatial data
# Add in WHO_REGION column based on matches to the shapes file
shp = gpd.read_file("data/shp_africa_adm0.geojson")
shp = shp[["WHO_REGION", "ADM0_NAME", "ISO_3_CODE"]]
# Filter the age DataFrame based on the Location column having a match in the shp ADM0_NAME column
age = age[age["ISO_3_CODE"].isin(shp["ISO_3_CODE"])]
# Merge the age DataFrame with the shp GeoDataFrame on the ADM0_NAME column
age = age.merge(shp[["ISO_3_CODE", "ADM0_NAME"]], on="ISO_3_CODE", how="left")

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
        "ADM0_NAME",
        "ISO_3_CODE",
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
age = age.melt(id_vars=["ADM0_NAME", "ISO_3_CODE", "Year"], var_name="age_group", value_name="population")
age = age.reset_index(drop=True)
print(age.head())

### Export the curated data
age.to_csv("data/age_africa.csv", index=False)

print("Done.")
