import re

import pandas as pd
from unidecode import unidecode

# DATA SOURCE: https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Most%20used

# Define file path (modify as needed)
file_path = "data/curation_scripts/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx"  # Corrected file path

# Load the file while skipping the first 15 rows and using the 16th row as headers
df = pd.read_excel(file_path, skiprows=16)
df = df.clean_names()

# Display the column names to identify available options
print("Columns available:", df.columns.tolist())

# Select and rename specific columns (Modify column names based on your dataset)
selected_columns = {
    "region_subregion_country_or_area_*": "region",
    "iso3_alpha_code": "iso",
    "year": "year",
    "total_population_as_of_1_january_thousands_": "tot_pop",
    "crude_birth_rate_births_per_1_000_population_": "cbr",
}

# Extract and rename the relevant columns
df_selected = df[selected_columns.keys()].rename(columns=selected_columns)

# Clean the region names by removing non-letter characters and converting letters with accents
df_selected["region"] = df_selected["region"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", unidecode(x)))

# Filter the data to countries in Africa
isos = [
    "DZA",
    "AGO",
    "BEN",
    "BWA",
    "BFA",
    "BDI",
    "CPV",
    "CMR",
    "CAF",
    "TCD",
    "COM",
    "COG",
    "COD",
    "DJI",
    "EGY",
    "GNQ",
    "ERI",
    "SWZ",
    "ETH",
    "GAB",
    "GMB",
    "GHA",
    "GIN",
    "GNB",
    "CIV",
    "KEN",
    "LSO",
    "LBR",
    "LBY",
    "MDG",
    "MWI",
    "MLI",
    "MRT",
    "MUS",
    "MYT",
    "MAR",
    "MOZ",
    "NAM",
    "NER",
    "NGA",
    "RWA",
    "STP",
    "SEN",
    "SYC",
    "SLE",
    "SOM",
    "ZAF",
    "SSD",
    "SDN",
    "TGO",
    "TUN",
    "UGA",
    "TZA",
    "ESH",
    "ZMB",
    "ZWE",
]
df_filtered = df_selected[df_selected["iso"].isin(isos)]
assert len(df_filtered["iso"].unique()) == len(isos)

# Filter by year (2010 - 2035)
df_filtered = df_filtered[(df_filtered["year"] >= 2010) & (df_filtered["year"] <= 2035)]

# Display the first few rows of the selected data
print(df_filtered.head())

# Save the cleaned dataset (optional)
df_filtered.to_csv("data/unwpp.csv", index=False)
