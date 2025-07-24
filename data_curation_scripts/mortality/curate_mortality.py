import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

### Import data
mortality_path = "data_curation_scripts/mortality/IHME-GBD_2021_DATA-026d71a8-1.csv"
mortality = pd.read_csv(mortality_path)

print("Original mortality data shape:", mortality.shape)
print("Columns:", mortality.columns.tolist())
print("\nFirst few rows:")
print(mortality.head())


### Curate the age data
# Convert age groups to numeric ranges for easier processing
def parse_age_group(age_str):
    """Parse age group string to min and max ages"""
    if age_str == "<5 years":
        return 0, 4
    elif age_str == "95+ years":
        return 95, 100  # Use 100 as max for 95+
    else:
        # Handle "5-9 years", "10-14 years", etc.
        parts = age_str.replace(" years", "").split("-")
        return int(parts[0]), int(parts[1])


# Apply parsing
age_parsed = mortality["age"].apply(parse_age_group)
mortality["age_min"] = [x[0] for x in age_parsed]
mortality["age_max"] = [x[1] for x in age_parsed]

# Select relevant columns
mortality = mortality[["age_min", "age_max", "year", "val"]]

# Rename columns to match your naming conventions
mortality = mortality.rename(columns={"val": "deaths_per_100k"})


### Convert rates to probabilities
# IHME provides rates per 100,000 population
# Convert to daily death probabilities for simulation use
def rate_to_daily_probability(rate_per_100k, age_min, age_max):
    """Convert annual mortality rate per 100k to daily death probability"""
    # Annual rate per person
    annual_rate_per_person = rate_per_100k / 100000

    # Convert to daily probability (assuming constant rate over the year)
    daily_prob = 1 - np.exp(-annual_rate_per_person / 365.25)

    return daily_prob


mortality["daily_death_prob"] = mortality.apply(
    lambda row: rate_to_daily_probability(row["deaths_per_100k"], row["age_min"], row["age_max"]), axis=1
)

# Paste together age_min and age_max to get the age group
mortality["age"] = mortality["age_min"].astype(str) + "-" + mortality["age_max"].astype(str)
mortality.loc[mortality["age"] == "95-100", "age"] = "95-99"


print("\nFinal curated data shape:", mortality.shape)
print("\nSample of curated data:")
print(mortality.head(10))

### Export the curated data
mortality.to_csv("data/mortality_Nigeria_2021.csv", index=False)

print("\nDone. Curated mortality data exported to data/mortality_curated.csv")
