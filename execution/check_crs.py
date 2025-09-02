# check_crs.py
import sys
import rasterio
import geopandas as gpd

if len(sys.argv) != 3:
    print("Usage: python check_crs.py <path_to_raster.tif> <path_to_shapefile.shp>")
    sys.exit(1)

raster_path = sys.argv[1]
shapefile_path = sys.argv[2]

try:
    print("-" * 50)
    # Check Raster CRS
    with rasterio.open(raster_path) as src:
        print(f"RASTER CRS: {src.crs}")
        raster_crs = src.crs

    # Check Shapefile CRS
    gdf = gpd.read_file(shapefile_path)
    print(f"SHAPEFILE CRS: {gdf.crs}")
    shapefile_crs = gdf.crs

    # Compare them
    print("-" * 50)
    if raster_crs == shapefile_crs:
        print("✅ SUCCESS: The Coordinate Reference Systems match.")
    else:
        print("❌ ERROR: The Coordinate Reference Systems DO NOT match.")
    print("-" * 50)

except Exception as e:
    print(f"An error occurred: {e}")