from pathlib import Path

import geopandas as gpd
import pandas as pd
import rastertools
from PIL import Image
from rastertools import raster_clip

print(f"rastertools version v{rastertools.__version__}")

# Increase the MAX_IMAGE_PIXELS limit to handle large images
Image.MAX_IMAGE_PIXELS = None


# Load the polis_adm2_africa.shp file
shapefile_path = "data/curation_scripts/shapes/polis/polis_adm2_africa.shp"
gdf = gpd.read_file(shapefile_path)

# Extract all of the unique ISO codes
isos = gdf["is_3_cd"].unique()

# Clip the WorldPop raster with the shapefile for each ISO code
n_successes = 0
n_fails = 0
failed_isos = []
for iso in isos:
    print(f"Processing raster for {iso}")
    shape_file = Path(f"data/curation_scripts/shapes/polis/polis_adm2_{iso}.shp")
    raster_file = Path(f"data/curation_scripts/pop/worldpop_rasters/{iso}_ppp_2020.tif")
    output_file = f"data/curation_scripts/pop/worldpop_rasters/pop_adm2_{iso}.csv"
    if not Path(output_file).exists:
        try:
            # Clipping raster with shapes (only pop values)
            pop_dict = raster_clip(raster_file, shape_file, include_latlon=True, shape_attr="dot_name", quiet=True)
            n_successes += 1

            # Save the popdict as a csv
            pop_df = pd.DataFrame.from_dict(pop_dict, orient="index")
            pop_df.reset_index(inplace=True)
            pop_df.rename(columns={"index": "dot_name"}, inplace=True)
            pop_df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"Failed to process data for {iso}: {e}")
            n_fails += 1
            failed_isos.append(iso)

    else:
        print(f"File already exists for {iso}: {output_file}")
        n_successes += 1

# Print summary
print(f"\nWorldpop rasters successfully downloaded or already existed: {n_successes} / {len(isos)}")
print(f"Worldpop rasters that couldn't be downloaded: {n_fails} / {len(isos)}")
if failed_isos:
    print("List of ISO codes that couldn't be downloaded:")
    for iso in failed_isos:
        print(iso)

print("Done.")


# iso = 'BEN'

# # Using example DRC shapefile and raster
# shape_file = Path(f'data/curation_scripts/shapes/polis/polis_adm2_{iso}.shp')
# raster_file = Path(f'data/curation_scripts/pop/worldpop_rasters/{iso}_ppp_2020.tif')


# Save the popdict as a JSON file
# utils.save_json(popdict, json_path=f"data/curation_scripts/pop/worldpop_rasters/pop_adm2_{iso}.json", sort_keys=True)


# from math import log10
# y = [e['lat'] for e in popdict.values()]
# x = [e['lon'] for e in popdict.values()]
# c = [5*log10(e['pop']+0.1) for e in popdict.values()]
# fig, ax = plot_shapes(shape_file, linewidth=0.2, alpha=1.0, edgecolor='k', facecolor='None')
# ax.scatter(x,y,s=c, alpha=0.3)
# ax.set_box_aspect(1)


# # Directory containing the individual shapefiles and WorldPop raster TIFFs
# shapefile_dir = 'data/curation_scripts/shapes/polis'
# raster_dir = 'data/curation_scripts/pop/worldpop_rasters'

# # Iterate over each unique ISO code
# for iso in isos:
#     print(f"Processing ISO code: {iso}")

#     # Load the shapefile with the current ISO code
#     iso_shapefile_path = os.path.join(shapefile_dir, f"adm2_{iso}.shp")
#     if os.path.exists(iso_shapefile_path):
#         iso_gdf = gpd.read_file(iso_shapefile_path)
#         print(f"Loaded shapefile for ISO code: {iso}")
#     else:
#         print(f"Shapefile for ISO code {iso} does not exist.")
#         continue

#     # Load the WorldPop raster TIFF with the current ISO code
#     raster_path = os.path.join(raster_dir, f"{iso}_ppp_2020.tif")
#     if os.path.exists(raster_path):
#         with rasterio.open(raster_path) as src:
#             raster_data = src.read(1)  # Read the first band
#             print(f"Loaded raster for ISO code: {iso}")
#     else:
#         print(f"Raster for ISO code {iso} does not exist.")
#         continue

#     # Perform any additional processing here
#     # ...
