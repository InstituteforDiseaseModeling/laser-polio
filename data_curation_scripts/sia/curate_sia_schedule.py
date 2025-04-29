import pandas as pd

from laser_polio.utils import clean_strings

df = pd.read_csv("data/curation_scripts/vx/scenario_sia_1_2024-10-28.csv", index_col=0)

# Curate the admin names & create a dot_name for each shape
columns_to_clean = ["adm0_name", "adm1_name", "adm2_name"]
df[columns_to_clean] = df[columns_to_clean].map(clean_strings)
df["dot_name"] = df.apply(lambda row: f"{'AFRO'}:{row['adm0_name']}:{row['adm1_name']}:{row['adm2_name']}", axis=1)

# Select only the start_date and dot_name columns
df = df[["start_date", "dot_name"]]

# Rename start_date to date
df = df.rename(columns={"start_date": "date"})

# Add age ranges
df["age_min"] = 0
df["age_max"] = 5 * 365

# Reset the index
df = df.reset_index(drop=True)

# Save the curated SIA schedule to a CSV file
df.to_csv("data/sia_scenario_1.csv", index=False)

print("Done!")
