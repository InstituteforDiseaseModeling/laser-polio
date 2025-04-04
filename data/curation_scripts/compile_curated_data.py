import geopandas as gpd
import pandas as pd

pd.set_option("display.max_columns", None)


def check_duplicates(df, subset):
    df_name = next(name for name in globals() if globals()[name] is df)  # Get variable name
    if df.duplicated(subset=subset).sum() > 0:
        print(f"There are duplicates in the {df_name} dataset.")
        # Print the rows with duplicates
        duplicates = df[df.duplicated(subset=subset, keep=False)]
        print(duplicates)
    else:
        print(f"The {df_name} dataset has no duplicates.")


# Load the datasets
cbr = pd.read_csv("data/curation_scripts/cbr/cbr_curated.csv")
pop = pd.read_csv("data/curation_scripts/pop/dpt_district_summaries_curated.csv")
pop = pop[["adm0_name", "adm1_name", "adm2_name", "dot_name", "year", "under5_pop"]]
ri = pd.read_csv("data/curation_scripts/ri/ri_curated.csv")
re = pd.read_csv("data/curation_scripts/random_effects/random_effects_curated.csv")
# sia = pd.read_csv("data/curation_scripts/sia/sia_random_effects_curated.csv")
underwt = pd.read_csv("data/curation_scripts/individual_risk/underwt_u5_2019.csv")
# Load the shapefiles for validation purposes
shp0 = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")  # gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")
shp1 = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm1")  # gpd.read_file("data/shp_africa_adm1.geojson")
shp2 = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")  # gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")
n_adm0 = len(shp0)
n_adm1 = len(shp1)
n_adm2 = len(shp2)

# print(cbr.head())
# print(pop.head())
# print(ri.head())
# print(sia.head())
# print(underwt.head())

# # Print lengths of datasets
# print(f'Length of cbr: {len(cbr)}')
# print(f'Length of pop: {len(pop)}')
# print(f'Length of ri: {len(ri)}')
# print(f'Length of sia: {len(sia)}')
# print(f'Length of underwt: {len(underwt)}')

### Validate the shp data
len(shp2["dot_name"].unique())
len(shp2["guid"].unique())
assert len(shp2["dot_name"].unique()) == len(shp2["guid"].unique()), (
    "The number of unique dot_name and guid are not equal in the shp2 dataset."
)

### Validate the cbr dataset
# print(cbr.head())
# cbr should have values for each country and year
n_years_cbr = len(cbr["year"].unique())
assert len(cbr) == n_adm0 * n_years_cbr, "The cbr dataset does not have values for each country and year."
n_na_cbr = cbr.isna().sum()
assert n_na_cbr.sum() == 0, "There are missing values in the cbr dataset. Please check the data sources and curate the missing values."
# Check for duplicates
check_duplicates(cbr, ["adm0_name", "year"])


### Validate the pop dataset
# print(pop.head())
# Ensure that there are no missing values in the pop dataset
n_na_pop = pop.isna().sum()
assert n_na_pop.sum() == 0, "There are missing values in the pop dataset. Please check the data sources and curate the missing values."
# Ensure that every dot_name in the shp2 dataset is in the pop dataset
assert shp2["dot_name"].isin(pop["dot_name"]).all(), "Not every dot_name in the shp2 dataset is in the pop dataset."
# Warn if the pop dataset has extra dot_names
extra_dot_names = pop[~pop["dot_name"].isin(shp2["dot_name"])]["dot_name"].unique()
if len(extra_dot_names) > 0:
    print(
        f"Warning: There are {len(extra_dot_names)} extra dot_names in the pop dataset that are not in the shp2 dataset. These extra dot_names will be dropped."
    )
    # Drop any dot_names that aren't in the shp2 dataset
    pop = pop[pop["dot_name"].isin(shp2["dot_name"])]
# Check for duplicates
check_duplicates(pop, ["adm0_name", "adm1_name", "adm2_name", "dot_name", "year"])


### Validate the ri dataset
# print(ri.head())
# Ensure that there are no missing values in the ri dataset
n_na_ri = ri.isna().sum()
assert n_na_ri.sum() == 0, "There are missing values in the ri dataset. Please check the data sources and curate the missing values."
# Ensure that every dot_name in the shp2 dataset is in the ri dataset
assert shp2["dot_name"].isin(ri["dot_name"]).all(), "Not every dot_name in the shp2 dataset is in the ri dataset."
# Warn if the ri dataset has extra dot_names
extra_dot_names = ri[~ri["dot_name"].isin(shp2["dot_name"])]["dot_name"].unique()
if len(extra_dot_names) > 0:
    print(
        f"Warning: There are {len(extra_dot_names)} extra dot_names in the ri dataset that are not in the shp2 dataset. These extra dot_names will be dropped."
    )
    # Drop any dot_names that aren't in the shp2 dataset
    ri = ri[ri["dot_name"].isin(shp2["dot_name"])]
# Check for duplicates
check_duplicates(ri, ["adm0_name", "adm1_name", "adm2_name", "dot_name", "year"])


### Validate the re (random effects for R_eff & SIA) dataset
# print(re.head())
# re should have values for each ADM1
assert len(re) == n_adm1, "The re dataset does not have values for each ADM1."
# Check for duplicates
check_duplicates(re, ["adm0_name", "adm1_name"])


# ### Validate the sia dataset
# print(sia.head())
# # sia should have values for each ADM1
# assert len(sia) == n_adm1, "The sia dataset does not have values for each ADM1."
# # Check for duplicates
# check_duplicates(sia, ["adm0_name", "adm1_name"])


# ### Validate the underwt dataset
# # print(underwt.head())
# # Ensure that there are no missing values in the underwt dataset
# n_na_underwt = underwt["prop_underwt"].isna().sum()
# assert n_na_underwt.sum() == 0, (
#     "There are missing values in the underwt dataset. Please check the data sources and curate the missing values."
# )
# # Ensure that every dot_name in the shp2 dataset is in the underwt dataset
# assert shp2["dot_name"].isin(underwt["dot_name"]).all(), "Not every dot_name in the shp2 dataset is in the underwt dataset."
# # Warn if the underwt dataset has extra dot_names
# extra_dot_names = underwt[~underwt["dot_name"].isin(shp2["dot_name"])]["dot_name"].unique()
# if len(extra_dot_names) > 0:
#     print(
#         f"Warning: There are {len(extra_dot_names)} extra dot_names in the underwt dataset that are not in the shp2 dataset. These extra dot_names will be dropped."
#     )
#     # Drop any dot_names that aren't in the shp2 dataset
#     underwt = underwt[underwt["dot_name"].isin(shp2["dot_name"])]
# # Check for duplicates
# check_duplicates(underwt, ["dot_name"])


### Merge the datasets
df = cbr.merge(pop, on=["adm0_name", "year"])
df = df.merge(ri, on=["adm0_name", "adm1_name", "adm2_name", "dot_name", "year"])
df = df.merge(re, on=["adm0_name", "adm1_name"], how="left")
# df = df.merge(sia, on=["adm0_name", "adm1_name"], how="left")
# df = df.merge(underwt[["dot_name", "prop_underwt"]], on="dot_name", how="left")
# Reorder
df = df[
    [
        "adm0_name",
        "adm1_name",
        "adm2_name",
        "dot_name",
        "year",
        "cbr",
        "under5_pop",
        "immunity_ri_nOPV2",
        "reff_random_effect",
        "sia_random_effect",
    ]
]
print(df.head())
print(f"Length of df: {len(df)}")


### Validate the merged dataset
# Ensure that there are no missing values in the merged dataset
n_na_df = df.isna().sum()
assert n_na_df.sum() == 0, "There are missing values in the merged dataset. Please check the data sources and curate the missing values."
# Ensure that every dot_name in the shp2 dataset is in the merged dataset
assert shp2["dot_name"].isin(df["dot_name"]).all(), "Not every dot_name in the shp2 dataset is in the merged dataset."
# Ensure that the merged dataset is the expected length
n_years_df = len(df["year"].unique())
assert len(df) == n_adm2 * n_years_cbr, "The merged dataset does not have values for each dot_name and year."


### Tidy up
df = df.copy()  # Create a copy of the DataFrame
df.rename(columns={"under5_pop": "pop_u5", "immunity_ri_nOPV2": "ri_eff"}, inplace=True)
df.head()


### Save the full curated dataframe
df.to_csv("data/compiled_cbr_pop_ri_sia_underwt_africa.csv", index=False)


print("Done.")
