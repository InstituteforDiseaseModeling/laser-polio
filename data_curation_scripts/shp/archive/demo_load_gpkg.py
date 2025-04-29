from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt


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
    "adm0": "data/curation_scripts/shp/shape0.gpkg",
    "adm1": "data/curation_scripts/shp/shape1.gpkg",
    "adm2": "data/curation_scripts/shp/shape2.gpkg",
}
output_gpkg = Path("data/curation_scripts/shp/shape_combined_simplified.gpkg")
output_dir = Path("data/curation_scripts/shp/plots")
tolerance = 0.01  # ≈ 1 km if CRS is WGS84
# --------------------------

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
