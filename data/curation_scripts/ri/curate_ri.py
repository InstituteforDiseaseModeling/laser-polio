import pandas as pd

"""
This code curates the routine immunization (RI) data for Africa. The data is from the DPT district summaries curated dataset
from the polio-immunity-mapping repo. The immunity_ri_nOPV2 value represents the expected proportion of childrent to be immune
to type 2 poliovirus from routine administration of nOPV2. This combines 1, 2, and 3 doses of DPT vaccine with an efficacy estimated
nOPV2 trials in Gambia & Bangladesh. See the nOPV2_does_efficacy_notes_20250107.pdf for more details.
"""

# Load the data
ri = pd.read_csv("data/curation_scripts/pop/dpt_district_summaries_curated.csv")

# Convert column names to lowercase
ri.columns = ri.columns.str.lower()

# Filter for Nigeria and the latest year
latest_year = ri["year"].max()

# Rename column
ri = ri.rename(columns={"dpt3": "dpt3_orig"})

# Modify dpt values
ri["dpt3"] = ri[["dpt1", "dpt3_orig"]].min(axis=1)
ri["dpt2"] = (ri["dpt1"] + ri["dpt3"]) / 2

# Select relevant columns
ri = ri[[col for col in ri.columns if "adm" in col] + ["dot_name", "year"] + [col for col in ri.columns if "dpt" in col]]

# Table 3 of Gambia
nOPV2 = 431 / 646
bOPV = 25 / 95
nOPV2_child = 183 / 248
bOPV_child = 47 / 252

VE = 1 - (1 - nOPV2) * (1 - bOPV)
VE_perdose = 1 - (1 - VE) ** (1 / 2)

# Immunogenicity per dose
perdose = 0.389  # old, 40

# Compute immunity_ri_nOPV2
ri.loc[:, "immunity_ri_nOPV2"] = (
    (ri["dpt1"] - ri["dpt2"]) * perdose + (ri["dpt2"] - ri["dpt3"]) * (1 - (1 - perdose) ** 2) + ri["dpt3"] * (1 - (1 - perdose) ** 3)
)

print(ri.head())

# Round the immunity_ri_nOPV2 column to 3 decimal places
ri["immunity_ri_nOPV2"] = ri["immunity_ri_nOPV2"].round(3)

# # Filter to the latest year
# ri = ri[ri['year'] == latest_year]

# Count number of NA values
n_na = ri["immunity_ri_nOPV2"].isna().sum()
if n_na > 0:
    print(
        f"There are {n_na} missing values in the immunity_ri_nOPV2 column. Filling in missing values with the mean from higher admin levels."
    )

# Fill in missing values with the mean from higher admin levels
ri.loc[:, "immunity_ri_nOPV2"] = ri.groupby(["adm0_name", "adm1_name", "year"])["immunity_ri_nOPV2"].transform(lambda x: x.fillna(x.mean()))
ri.loc[:, "immunity_ri_nOPV2"] = ri.groupby(["adm0_name", "year"])["immunity_ri_nOPV2"].transform(lambda x: x.fillna(x.mean()))

# Check for missing values
assert ri["immunity_ri_nOPV2"].isna().sum() == 0, (
    "There are still missing values in the immunity_ri_nOPV2 column after taking the adm1 or adm0 mean."
)

# Select the relevant columns
ri = ri[["adm0_name", "adm1_name", "adm2_name", "dot_name", "year", "immunity_ri_nOPV2"]]
ri = ri.rename(columns={"adm0_name": "adm0_name", "adm1_name": "adm1_name", "adm2_name": "adm2_name"})
print(ri.head())
ri.to_csv("data/curation_scripts/ri/ri_curated.csv", index=False)

# # Convert the DataFrame to a dictionary
# ri_dict = ri.set_index('dot_name')['immunity_ri_nOPV2'].to_dict()

# # Save the dictionary as a JSON file
# with open("data/ri_africa.json", "w") as json_file:
#     json.dump(ri_dict, json_file, indent=4)

print("Done")
