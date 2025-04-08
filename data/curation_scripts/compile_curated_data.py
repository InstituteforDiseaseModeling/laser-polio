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
demog = pd.read_csv("data/curation_scripts/pop/demog_data_POLIS_ADM02_yearly_cbr.csv")  # cbr & pop
ri = pd.read_csv("data/curation_scripts/ri/ri_curated.csv")
re = pd.read_csv("data/curation_scripts/random_effects/random_effects_curated.csv")
underwt = pd.read_csv("data/curation_scripts/individual_risk/underwt_u5_2019.csv")

# Load the shapefiles for validation purposes
shp0 = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")  # gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm0")
shp1 = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm1")  # gpd.read_file("data/shp_africa_adm1.geojson")
shp2 = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")  # gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")
n_adm0 = len(shp0)
n_adm1 = len(shp1)
n_adm2 = len(shp2)

# print(cbr.head())
# print(demog.head())
# print(ri.head())
# print(sia.head())
# print(underwt.head())

# # Print lengths of datasets
# print(f'Length of cbr: {len(cbr)}')
# print(f'Length of demog: {len(demog)}')
# print(f'Length of ri: {len(ri)}')
# print(f'Length of sia: {len(sia)}')
# print(f'Length of underwt: {len(underwt)}')

### Validate the shp data
len(shp2["dot_name"].unique())
len(shp2["guid"].unique())
assert len(shp2["dot_name"].unique()) == len(shp2["guid"].unique()), (
    "The number of unique dot_name and guid are not equal in the shp2 dataset."
)


### Validate the demog dataset with cbr & demog
# print(demog.head())
# Ensure that there are no missing values in the demog dataset
n_na_demog = demog.isna().sum()
assert n_na_demog.sum() == 0, "There are missing values in the demog dataset. Please check the data sources and curate the missing values."
# Ensure that every dot_name in the shp2 dataset is in the demog dataset
assert shp2["dot_name"].isin(demog["dot_name"]).all(), "Not every dot_name in the shp2 dataset is in the demog dataset."
# Warn if the demog dataset has extra dot_names
extra_dot_names = demog[~demog["dot_name"].isin(shp2["dot_name"])]["dot_name"].unique()
if len(extra_dot_names) > 0:
    print(
        f"Warning: There are {len(extra_dot_names)} extra dot_names in the demog dataset that are not in the shp2 dataset. These extra dot_names will be dropped."
    )
    # Drop any dot_names that aren't in the shp2 dataset
    demog = demog[demog["dot_name"].isin(shp2["dot_name"])]
# Check for duplicates
check_duplicates(demog, ["adm0_name", "adm1_name", "adm2_name", "dot_name", "year"])


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


### Validate the underwt dataset
# print(underwt.head())
# Ensure that there are no missing values in the underwt dataset
n_na_underwt = underwt["prop_underwt"].isna().sum()
assert n_na_underwt.sum() == 0, (
    "There are missing values in the underwt dataset. Please check the data sources and curate the missing values."
)
# Ensure that every dot_name in the shp2 dataset is in the underwt dataset
# assert shp2["dot_name"].isin(underwt["dot_name"]).all(), "Not every dot_name in the shp2 dataset is in the underwt dataset."
# Warn if the underwt dataset has extra dot_names
extra_dot_names = underwt[~underwt["dot_name"].isin(shp2["dot_name"])]["dot_name"].unique()
if len(extra_dot_names) > 0:
    print(
        f"Warning: There are {len(extra_dot_names)} extra dot_names in the underwt dataset that are not in the shp2 dataset. These extra dot_names will be dropped."
    )
    # Drop any dot_names that aren't in the shp2 dataset
    underwt = underwt[underwt["dot_name"].isin(shp2["dot_name"])]
# Check for duplicates
check_duplicates(underwt, ["dot_name"])


### Merge the datasets
df = demog.copy()
df = df.merge(ri, on=["adm0_name", "adm1_name", "adm2_name", "dot_name", "year"], how="left")
df = df.merge(re, on=["adm0_name", "adm1_name"], how="left")
df = df.merge(underwt[["dot_name", "prop_underwt"]], on="dot_name", how="left")
# Reorder
df = df[
    [
        "adm0_name",
        "adm1_name",
        "adm2_name",
        "dot_name",
        "year",
        "cbr",
        "pop_total",
        "immunity_ri_nOPV2",
        "reff_random_effect",
        "sia_random_effect",
        "prop_underwt",
    ]
]
print(df.head())
print(f"Length of df: {len(df)}")


# Fill in missing values with the ri data cuz it's only for the latest year
df.loc[:, "immunity_ri_nOPV2"] = df.groupby(["adm0_name", "adm1_name", "adm2_name"])["immunity_ri_nOPV2"].transform(
    lambda x: x.fillna(x.mean())
)
df.loc[:, "prop_underwt"] = df.groupby(["adm0_name", "adm1_name"])["immunity_ri_nOPV2"].transform(lambda x: x.fillna(x.mean()))


### Validate the merged dataset
# Ensure that there are no missing values in the merged dataset
n_na_df = df.isna().sum()
assert n_na_df.sum() == 0, "There are missing values in the merged dataset. Please check the data sources and curate the missing values."
# Ensure that every dot_name in the shp2 dataset is in the merged dataset
assert shp2["dot_name"].isin(df["dot_name"]).all(), "Not every dot_name in the shp2 dataset is in the merged dataset."
# Ensure that the merged dataset is the expected length
n_years_df = len(df["year"].unique())
n_years_demog = len(demog["year"].unique())
assert len(df) == n_adm2 * n_years_demog, "The merged dataset does not have values for each dot_name and year."


### Tidy up
df = df.copy()  # Create a copy of the DataFrame
df.rename(columns={"immunity_ri_nOPV2": "ri_eff"}, inplace=True)
df.head()


### Save the full curated dataframe
df.to_csv("data/compiled_cbr_pop_ri_sia_underwt_africa.csv", index=False)


print("Done.")
