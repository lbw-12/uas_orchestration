# check_bounds.py
import sys
import rasterio
import geopandas as gpd
from shapely.geometry import box

if len(sys.argv) != 3:
    print("Usage: python check_bounds.py <path_to_raster.tif> <path_to_shapefile.shp>")
    sys.exit(1)

raster_path = sys.argv[1]
shapefile_path = sys.argv[2]

try:
    print("-" * 50)
    # Get Raster Bounds
    with rasterio.open(raster_path) as src:
        raster_bounds = src.bounds
        raster_geom = box(*raster_bounds)
        print(f"RASTER Bounds: {raster_bounds}")

    # Get Shapefile Bounds
    gdf = gpd.read_file(shapefile_path)
    shapefile_bounds = gdf.total_bounds
    print(f"SHAPEFILE Bounds: {shapefile_bounds}")

    # Check for intersection
    print("-" * 50)
    if raster_geom.intersects(gdf.unary_union):
        print("✅ SUCCESS: The geometries intersect according to the libraries.")
    else:
        print("❌ ERROR: The geometries DO NOT intersect according to the libraries.")
    print("-" * 50)

except Exception as e:
    print(f"An error occurred: {e}")