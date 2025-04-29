# Load the shapes2.geojson file and extract the centroids of the shapes.


import geopandas as gpd
import numpy as np
import pandas as pd
from alive_progress import alive_bar

from laser_polio.utils import clean_strings


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
    shapes = gpd.read_file("data/curation_scripts/shapes/africa_polis_adm2.geojson")

    # Curate the admin names & create a dot_name for each shape
    columns_to_clean = ["who_region", "adm0_name", "adm1_name", "adm2_name"]
    shapes[columns_to_clean] = shapes[columns_to_clean].map(clean_strings)
    shapes["dot_name"] = shapes.apply(lambda row: f"{row['who_region']}:{row['adm0_name']}:{row['adm1_name']}:{row['adm2_name']}", axis=1)

    # Filter the shapes file to entries where adm0_name == 'NIGERIA'
    # shapes = shapes[shapes['dot_name'].str.contains('NIGERIA', case=False, na=False)]
    # Reset the index to ensure it is a simple range index
    # shapes = shapes.reset_index(drop=True)

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(shapes)

    # Create a DataFrame with country names as row and column labels
    dot_names = shapes["dot_name"]
    distance_df = pd.DataFrame(distance_matrix, index=dot_names, columns=dot_names)
    distance_df.head()

    # Save the distance matrix to a CSV file
    distance_df.to_csv("data/distance_matrix_africa_polis_adm2.csv")

    # # Print the list of dot_names containing "ZAMFARA"
    # zamfara_dot_names = shapes[shapes['dot_name'].str.contains("ZAMFARA", case=False, na=False)]['dot_name']
    # print("List of dot_names containing 'ZAMFARA':")
    # print(zamfara_dot_names.tolist())

    print("Done.")
