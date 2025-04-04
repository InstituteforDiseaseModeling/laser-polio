from datetime import datetime
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import sciris as sc

from laser_polio.utils import clean_strings

###################################
######### USER PARAMETERS #########

# Define the shapefiles and associated columns
shapefiles = [
    ("data/curation_scripts/shp/shape0.gpkg", ["who_region", "adm0_name"], "adm0"),
    ("data/curation_scripts/shp/shape1.gpkg", ["who_region", "adm0_name", "adm1_name"], "adm1"),
    ("data/curation_scripts/shp/shape2.gpkg", ["who_region", "adm0_name", "adm1_name", "adm2_name"], "adm2"),
]
output_gpkg_path = Path("data/shp_africa_low_res.gpkg")
output_dir = Path("data/curation_scripts/shp/plots")
tolerance = 0.01  # ≈ 1 km if CRS is WGS84

######### END OF USER PARS ########
###################################

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)
# Remove existing combined file to avoid layer conflicts
output_gpkg_path.unlink(missing_ok=True)


def plot_and_save(gdf, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", linewidth=0.2)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=2000)
    plt.close()
    print(f"📍 Saved plot: {output_path}")


# Process each shapefile
for shapefile, columns_to_clean, adm_level in shapefiles:
    # ----- Load and curate the shapefiles -----
    sc.printcyan(f"\n📦 Processing: {shapefile} → layer '{adm_level}'")

    # Load the shapefile
    gdf = gpd.read_file(shapefile)

    # Clean the admin names
    gdf[columns_to_clean] = gdf[columns_to_clean].map(clean_strings)

    # Generate 'dot_name'
    gdf["dot_name"] = gdf[columns_to_clean].agg(":".join, axis=1)

    # Ensure that all shapes are current (i.e., enddate > today)
    # Preprocess enddate to replace large dates with an acceptable date
    # Function to apply to each entry
    def replace_invalid_date(x):
        # Define a proper datetime object
        acceptable_date = pd.Timestamp("2100-12-31", tz="UTC")
        if "9999" in str(x):
            return acceptable_date
        return x

    # Apply the function to the 'enddate' column
    gdf["enddate"] = gdf["enddate"].astype(str).apply(replace_invalid_date)

    # Convert enddate to datetime.date
    gdf["enddatev2"] = pd.to_datetime(gdf["enddate"], errors="coerce").dt.date
    today = datetime.today().date()
    assert all(gdf.enddatev2 > today), "Not all enddate values are greater than the current date."

    # Filter to the countries in Africa (who_region == 'AFRO' or iso_2_code in emro_africa_countries)
    emro_africa_countries = ["EG", "LY", "SD", "TN", "MA", "SO", "DJ"]  # List of African EMRO countries
    # Apply the filter to the GeoDataFrame
    filtered_gdf = gdf[(gdf["who_region"] == "AFRO") | (gdf["iso_2_code"].isin(emro_africa_countries))]

    # Filter to the columns we need
    columns_to_keep = [
        *columns_to_clean,
        "iso_2_code",
        "iso_3_code",
        "dot_name",
        "guid",
        "startdate",
        "enddatev2",
        "center_lon",
        "center_lat",
        "geometry",
    ]
    filtered_gdf = filtered_gdf[columns_to_keep]
    filtered_gdf["startdate"] = pd.to_datetime(filtered_gdf["startdate"], errors="coerce").dt.date
    filtered_gdf = filtered_gdf.rename(columns={"enddatev2": "enddate"})  # rename enddatev2 to enddate

    # Fixes for shapefiles with duplicate dot_names
    # Check if dot_name and year combinations are unique
    if len(filtered_gdf) != len(filtered_gdf[["dot_name"]].drop_duplicates()):
        print("Before manual fixes, the dot_name entries are not unique.")
        # Print the rows with duplicates
        duplicates = filtered_gdf[filtered_gdf.duplicated(subset=["dot_name"], keep=False)]
        print(duplicates)
    else:
        print("Before manual fixes, the dot_name entries are unique.")
    # Manually update dot_name for duplicates
    filtered_gdf = filtered_gdf[
        filtered_gdf["guid"] != "{EE73F3EA-DD35-480F-8FEA-5904274087C4}"
    ]  # Drop duplicate SOMALIA:LOWER_JUBA with earliest startdate
    filtered_gdf.loc[filtered_gdf["guid"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "dot_name"] = "AFRO:GUINEA_BISSAU:BAFATA:BAMBADINCA2"
    filtered_gdf.loc[filtered_gdf["guid"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "adm2_name"] = "BAMBADINCA2"
    # Check again for duplicates
    if len(filtered_gdf) != len(filtered_gdf[["dot_name"]].drop_duplicates()):
        print("After manual fixes, the dot_name entries are not unique.")
        # Print the rows with duplicates
        duplicates = filtered_gdf[filtered_gdf.duplicated(subset=["dot_name"], keep=False)]
        print(duplicates)
    else:
        print("After manual fixes, the dot_name entries are unique.")
    # # Print the dot_names containing 'BAMBADINCA'
    # print(filtered_gdf[filtered_gdf['dot_name'].str.contains('BAMBADINCA')])

    # ----- Now downsample those curated files & export them in geopackage format -----

    # Plot original
    plot_and_save(filtered_gdf, f"{adm_level.upper()} Original", output_dir / f"{adm_level}_original.png")

    # Simplify
    gdf_simplified = filtered_gdf.copy()
    gdf_simplified["geometry"] = filtered_gdf.geometry.simplify(tolerance=tolerance, preserve_topology=True)

    # Plot simplified
    plot_and_save(gdf_simplified, f"{adm_level.upper()} Simplified", output_dir / f"{adm_level}_simplified.png")

    # Ensure only one geometry column
    gdf_simplified = gdf_simplified[
        [col for col in gdf_simplified.columns if gdf_simplified[col].dtype.name != "geometry" or col == gdf_simplified.geometry.name]
    ]
    gdf_simplified.set_geometry("geometry", inplace=True)

    # Save simplified layer to combined GPKG
    gdf_simplified.to_file(output_gpkg_path, layer=adm_level, driver="GPKG")
    print(f"✅ Saved layer '{adm_level}' to {output_gpkg_path}")

    # Save the adm2 shape names
    if adm_level == "adm2":
        shp_names = filtered_gdf[["who_region", "adm0_name", "adm1_name", "adm2_name", "dot_name", "guid", "center_lon", "center_lat"]]
        shp_names.to_csv("data/shp_names_africa_adm2.csv", index=False)


# Show result
sc.printcyan("\n🗂 Final layers in combined GPKG:")
print(fiona.listlayers(output_gpkg_path))

sc.printcyan("Done.")
