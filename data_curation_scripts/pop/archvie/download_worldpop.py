import os

# List each of the unique iso3 codes in the africa_polis_adm0.shp file
import geopandas as gpd
from wpgpDownload.utils.convenience_functions import download_country_covariates as dl

shapefile_path = "data/curation_scripts/shapes/africa_polis_adm0.shp"
gdf = gpd.read_file(shapefile_path)
print(gdf.columns)
# List each of the unique ISO3 codes in the shapefile
isos = gdf["is_3_cd"].unique()


# Download the WorldPop population rasters for each of the unique ISO3 codes
output_dir = "data/curation_scripts/pop/worldpop_rasters"
os.makedirs(output_dir, exist_ok=True)
successful_downloads = 0
failed_downloads = 0
failed_isos = []
for iso in isos:
    print(iso)
    output_file = os.path.join(output_dir, f"{iso}_ppp_2020.tif")
    if not os.path.exists(output_file):
        try:
            dl(ISO=iso, out_folder="data/curation_scripts/pop/worldpop_rasters", prod_name="ppp_2020")
            successful_downloads += 1
        except Exception as e:
            print(f"Failed to download data for {iso}: {e}")
            failed_downloads += 1
            failed_isos.append(iso)
    else:
        print(f"File already exists for {iso}: {output_file}")
        successful_downloads += 1

# Print summary
print(f"\nWorldpop rasters successfully downloaded or already existed: {successful_downloads} / {len(isos)}")
print(f"Worldpop rasters that couldn't be downloaded: {failed_downloads} / {len(isos)}")
if failed_isos:
    print("List of ISO codes that couldn't be downloaded:")
    for iso in failed_isos:
        print(iso)

# # List the ISO codes of the countries available in the WorldPop dataset
# for iso in ISO_LIST:
#     print(iso)
