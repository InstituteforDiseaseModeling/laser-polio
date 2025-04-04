import os
import zipfile
from datetime import datetime
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from laser_polio.utils import clean_strings

# Path to the zipped folder containing GeoJSON files
zip_path = "data/curation_scripts/shp/geojson.zip"

# Directory within the zip archive containing the GeoJSON files
geojson_dir_in_zip = "scn/cvd2/results/geojson/"

# Directory to extract the GeoJSON files
extract_dir = "data/curation_scripts/shp/extracted_geojsons"
os.makedirs(extract_dir, exist_ok=True)

# Extract the GeoJSON files from the zip archive
extracted_geojson_files = [
    os.path.join(extract_dir, f.name) for f in Path(extract_dir).iterdir() if f.suffix == ".geojson"
]  # Check if the files have already been unzipped and saved
if not extracted_geojson_files:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # List all files in the zip archive
        all_files = zip_ref.namelist()
        # Filter for GeoJSON files in the specified directory
        geojson_files_in_zip = [f for f in all_files if f.startswith(geojson_dir_in_zip) and f.endswith(".geojson")]
        # Extract the filtered GeoJSON files
        for geojson_file in geojson_files_in_zip:
            # Extract the file to the specified directory
            file_name = os.path.basename(geojson_file)
            target_path = os.path.join(extract_dir, file_name)
            with zip_ref.open(geojson_file) as source, open(target_path, "wb") as target:
                target.write(source.read())
    # List all GeoJSON files in the extracted directory
    extracted_geojson_files = [os.path.join(extract_dir, f.name) for f in Path(extract_dir).iterdir() if f.suffix == ".geojson"]
# Define the shapefiles and associated columns
shapefiles = [
    ("countries.geojson", ["WHO_REGION", "ADM0_NAME"], "adm0"),
    ("provinces.geojson", ["WHO_REGION", "ADM0_NAME", "ADM1_NAME"], "adm1"),
    ("districts.geojson", ["WHO_REGION", "ADM0_NAME", "ADM1_NAME", "ADM2_NAME"], "adm2"),
]

# Process each shapefile
for shapefile, columns_to_clean, adm_level in shapefiles:
    # Load the shapefile
    geojson_file = next((f for f in extracted_geojson_files if shapefile in f), None)
    gdf = gpd.read_file(geojson_file)

    # Clean the admin names
    gdf[columns_to_clean] = gdf[columns_to_clean].map(clean_strings)

    # Generate 'dot_name'
    gdf["dot_name"] = gdf[columns_to_clean].agg(":".join, axis=1)

    # Ensure that all shapes are current (i.e., ENDDATE > today)
    # Preprocess ENDDATE to replace large dates with an acceptable date
    # Function to apply to each entry
    def replace_invalid_date(x):
        # Define a proper datetime object
        acceptable_date = pd.Timestamp("2100-12-31", tz="UTC")
        if "9999" in str(x):
            return acceptable_date
        return x

    # Apply the function to the 'ENDDATE' column
    gdf["ENDDATE"] = gdf["ENDDATE"].astype(str).apply(replace_invalid_date)

    # Convert ENDDATE to datetime.date
    gdf["ENDDATEv2"] = pd.to_datetime(gdf["ENDDATE"], errors="coerce").dt.date
    today = datetime.today().date()
    assert all(gdf.ENDDATEv2 > today), "Not all ENDDATE values are greater than the current date."

    # Filter to the countries in Africa (WHO_REGION == 'AFRO' or ISO_2_CODE in emro_africa_countries)
    emro_africa_countries = ["EG", "LY", "SD", "TN", "MA", "SO", "DJ"]  # List of African EMRO countries
    # Apply the filter to the GeoDataFrame
    filtered_gdf = gdf[(gdf["WHO_REGION"] == "AFRO") | (gdf["ISO_2_CODE"].isin(emro_africa_countries))]

    # Filter to the columns we need
    columns_to_keep = [
        *columns_to_clean,
        "ISO_2_CODE",
        "ISO_3_CODE",
        "dot_name",
        "GUID",
        "STARTDATE",
        "ENDDATEv2",
        "CENTER_LON",
        "CENTER_LAT",
        "geometry",
    ]
    filtered_gdf = filtered_gdf[columns_to_keep]
    filtered_gdf["STARTDATE"] = pd.to_datetime(filtered_gdf["STARTDATE"], errors="coerce").dt.date
    filtered_gdf = filtered_gdf.rename(columns={"ENDDATEv2": "ENDDATE"})  # rename enddatev2 to enddate

    # Fixes for shapefiles with duplicate dot_names
    # Check if dot_name and year combinations are unique
    if len(filtered_gdf) != len(filtered_gdf[["dot_name"]].drop_duplicates()):
        print("The dot_name entries are not unique.")
        # Print the rows with duplicates
        duplicates = filtered_gdf[filtered_gdf.duplicated(subset=["dot_name"], keep=False)]
        print(duplicates)
    else:
        print("The dot_name entries are unique.")
    # Manually update dot_name for duplicates
    filtered_gdf.loc[filtered_gdf["GUID"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "dot_name"] = "AFRO:GUINEA_BISSAU:BAFATA:BAMBADINCA2"
    filtered_gdf.loc[filtered_gdf["GUID"] == "{90211E77-0803-4728-B6CB-2F194DB4C21E}", "ADM2_NAME"] = "BAMBADINCA2"
    # Check again for duplicates
    if len(filtered_gdf) != len(filtered_gdf[["dot_name"]].drop_duplicates()):
        print("The dot_name entries are not unique.")
        # Print the rows with duplicates
        duplicates = filtered_gdf[filtered_gdf.duplicated(subset=["dot_name"], keep=False)]
        print(duplicates)
    else:
        print("The dot_name entries are unique.")
    # # Print the dot_names containing 'BAMBADINCA'
    # print(filtered_gdf[filtered_gdf['dot_name'].str.contains('BAMBADINCA')])

    # Save the curated shapefile as a GeoJSON file
    output_geojson_path = f"data/curation_scripts/shp/shp_africa_{adm_level}.geojson"
    filtered_gdf.to_file(output_geojson_path, driver="GeoJSON")

    # Save the adm2 shape names
    if adm_level == "adm2":
        shp_names = filtered_gdf[["WHO_REGION", "ADM0_NAME", "ADM1_NAME", "ADM2_NAME", "dot_name", "GUID", "CENTER_LON", "CENTER_LAT"]]
        shp_names.to_csv("data/shp_names_africa_adm2.csv", index=False)

    print(f"Curated shapefile saved as GeoJSON file: {output_geojson_path}")


# ----- Now downsample those curated files & export them in geopackage format -----


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


def simplify_and_save(input_path, layer_name, output_path_gpkg, output_dir, tolerance=0.01):
    print(f"\n📦 Processing: {input_path} → layer '{layer_name}'")

    # Load
    gdf = gpd.read_file(input_path)

    # Plot original
    plot_and_save(gdf, f"{layer_name.upper()} Original", output_dir / f"{layer_name}_original.png")

    # Simplify
    gdf_simplified = gdf.copy()
    gdf_simplified["geometry"] = gdf.geometry.simplify(tolerance=tolerance, preserve_topology=True)

    # Plot simplified
    plot_and_save(gdf_simplified, f"{layer_name.upper()} Simplified", output_dir / f"{layer_name}_simplified.png")

    # Ensure only one geometry column
    gdf_simplified = gdf_simplified[
        [col for col in gdf_simplified.columns if gdf_simplified[col].dtype.name != "geometry" or col == gdf_simplified.geometry.name]
    ]
    gdf_simplified.set_geometry("geometry", inplace=True)

    # Save simplified layer to combined GPKG
    gdf_simplified.to_file(output_path_gpkg, layer=layer_name, driver="GPKG")
    print(f"✅ Saved layer '{layer_name}' to {output_path_gpkg}")


# --------- Config ---------

input_files = {
    "adm0": "data/curation_scripts/shp/shp_africa_adm0.geojson",
    "adm1": "data/curation_scripts/shp/shp_africa_adm1.geojson",
    "adm2": "data/curation_scripts/shp/shp_africa_adm2.geojson",
}
output_gpkg = Path("data/shp_africa_low_res.gpkg")
output_dir = Path("data/curation_scripts/shp/plots")
tolerance = 0.01  # ≈ 1 km if CRS is WGS84

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Remove existing combined file to avoid layer conflicts
output_gpkg.unlink(missing_ok=True)

# Run simplification and plotting
for layer_name, file_path in input_files.items():
    simplify_and_save(file_path, layer_name, output_gpkg, output_dir, tolerance)

# Show result
print("\n🗂 Final layers in combined GPKG:")
print(fiona.listlayers(output_gpkg))

print("Done.")


# import matplotlib.pyplot as plt
# import random

# # Load the admin2 shapes
# shp = gpd.read_file('data/shp_africa_adm2.geojson')
# # Filter to ADM2_NAME containing BAMBADINCA
# shp_bam = shp[shp['ADM2_NAME'].str.contains('BAMBADINCA')]
# # Plot the filtered shapes
# shp_bam.plot()
# plt.title('Filtered Shapes Containing BAMBADINCA')
# plt.show()

# # Plot the filtered shapes with different fills and semi-transparency
# fig, ax = plt.subplots()
# colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(len(shp_bam))]
# shp_bam.plot(ax=ax, color=colors, alpha=0.5)

# # Add labels
# for idx, row in shp_bam.iterrows():
#     ax.annotate(text=row['dot_name'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
#                 xytext=(3, 3), textcoords='offset points', fontsize=8, color='black')

# # Add a legend
# handles = [plt.Line2D([0, 0], [0, 0], color=color, lw=4, label=dot_name) for color, dot_name in zip(colors, shp_bam['dot_name'])]
# ax.legend(handles=handles, title='dot_name', bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.title('Filtered Shapes Containing BAMBADINCA')
# plt.show()
