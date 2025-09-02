import geopandas
import argparse
import sys         # For printing errors to stderr

def count_point_geometries_in_shapefile(shapefile_path):
    """
    Counts the number of Point features and individual points within MultiPoint features
    in a shapefile.

    Args:
        shapefile_path (str): The path to the shapefile.

    Returns:
        int: The total number of individual points from Point or MultiPoint features.
             Returns None if an error occurs.
    """
    try:
        gdf = geopandas.read_file(shapefile_path)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return None

    total_individual_points = 0

    # Check if the geometry column exists
    if 'geometry' not in gdf.columns:
        print("Error: 'geometry' column not found in the shapefile.")
        return None

    for index, row in gdf.iterrows():
        geom = row['geometry']
        if geom is None:
            continue

        geom_type = geom.geom_type
        if geom_type == 'Point':
            total_individual_points += 1
        elif geom_type == 'MultiPoint':
            # A MultiPoint geometry is a collection of points.
            # geom.geoms gives an iterator of the individual Point objects.
            total_individual_points += len(geom.geoms)
        # Other geometry types (LineString, Polygon, etc.) are ignored.

    return total_individual_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count Point and MultiPoint features in a shapefile.")
    parser.add_argument("shapefile_path", type=str, help="Path to the .shp file")

    args = parser.parse_args()

    number_of_points = count_point_geometries_in_shapefile(args.shapefile_path)

    if number_of_points is not None:
        print(f"{number_of_points}")