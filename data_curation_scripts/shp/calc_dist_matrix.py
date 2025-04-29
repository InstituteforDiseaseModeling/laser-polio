# Load the shapes2.geojson file and extract the centroids of the shapes.
import geopandas as gpd
import numpy as np
import pandas as pd
from alive_progress import alive_bar

pd.set_option("display.max_columns", None)


def calculate_distance_matrix(gdf):
    """Calculates a distance matrix between polygon centroids in kilometers."""

    # Extract the centroids of the shapes
    gdf = gdf.to_crs("EPSG:3395")  # Convert to projected CRS for accurate distance calculation (Azimuthal Equidistant projection)
    gdf["centroid"] = gdf.geometry.centroid

    # Get centroids
    centroids = gdf.geometry.centroid

    # Create an empty distance matrix
    num_polygons = len(gdf)
    distance_matrix = np.zeros((num_polygons, num_polygons))

    # Calculate distances
    n_steps = num_polygons * num_polygons
    with alive_bar(n_steps, title="Progress:") as bar:
        for i in range(num_polygons):
            for j in range(num_polygons):
                distance_matrix[i, j] = centroids[i].distance(centroids[j]) / 1000  # Convert to km
                bar()  # Update the progress bar

    distance_matrix = np.round(distance_matrix)  # Round to nearest km

    return distance_matrix


if __name__ == "__main__":
    # Load the shapes2.geojson file
    try:
        shapes = gpd.read_file(filename="data/shp_africa_low_res.gpkg", layer="adm2")
    except Exception as e:
        print(f"Failed to read GeoJSON data with pyogrio: {e}")
        print("Trying to read with fiona...")

    # # Load the shapes2.geojson file
    # shapes = gpd.read_file('data/shp_africa_adm2.geojson')

    # Filter shapes to unique dot_names
    shapes = shapes.drop_duplicates(subset="dot_name")
    # Reset the index
    shapes.reset_index(drop=True, inplace=True)

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(shapes)

    # Create a DataFrame with country names as row and column labels
    dot_names = shapes["dot_name"]
    distance_df = pd.DataFrame(distance_matrix, index=dot_names, columns=dot_names)
    distance_df.head()
    df_backup = distance_df.copy()

    # Check for duplicate column names
    duplicates = distance_df.columns[distance_df.columns.duplicated()].unique()
    if len(duplicates) > 0:
        print(f"Duplicate column names found: {duplicates}")
    else:
        print("No duplicate column names found.")

    # Save the distance matrix
    distance_df.to_hdf("data/distance_matrix_africa_adm2.h5", key="dist_matrix", mode="w", complevel=9, complib="blosc")

    print("Done.")
