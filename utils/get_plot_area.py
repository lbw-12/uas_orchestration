import geopandas
import argparse
import warnings # To manage warnings from geopandas or shapely

def get_combined_polygon_area(shapefile_path): # Renamed for clarity as primary goal isn't always UTM conversion
    """
    Calculates the combined area of all Polygon and MultiPolygon features
    in a shapefile. If the shapefile's CRS is geographic (e.g., WGS84),
    it attempts to reproject the data to an appropriate UTM zone before
    calculating the area (results in square meters). Otherwise, area is
    calculated in the square units of the original projected CRS.

    Args:
        shapefile_path (str): The path to the .shp file.

    Returns:
        float: The total combined area.
               Returns None if an error occurs or no polygons are found.
        geopandas.CRS: The CRS object used for the area calculation.
                       Returns None if an error occurs.
        bool: True if reprojection to UTM occurred, False otherwise.
    """
    try:
        gdf = geopandas.read_file(shapefile_path)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return None, None, False

    if gdf.empty:
        print("Shapefile is empty.")
        return 0.0, gdf.crs, False # Return original CRS even if empty

    if 'geometry' not in gdf.columns or gdf['geometry'].isnull().all():
        print("Error: 'geometry' column not found or is empty in the shapefile.")
        return None, gdf.crs, False # Return original CRS

    original_crs = gdf.crs
    reprojected_to_utm = False
    crs_for_area_calculation = original_crs

    #print(f"Original Shapefile CRS: {original_crs.to_wkt(pretty=True) if original_crs else 'Unknown'}")
    #print("------------------------------")


    # Check if the original CRS is geographic
    if original_crs and original_crs.is_geographic:
        print(f"Original CRS is geographic ({original_crs.name}). Attempting to reproject to UTM for area calculation...")
        try:
            utm_crs_estimate = gdf.estimate_utm_crs(datum_name=original_crs.datum.name if original_crs.datum else None)

            if utm_crs_estimate:
                with warnings.catch_warnings(): # Suppress potential Shapely 2.0 warning about loss of Z
                    warnings.simplefilter("ignore", UserWarning)
                    gdf = gdf.to_crs(utm_crs_estimate)
                crs_for_area_calculation = gdf.crs # CRS after reprojection
                reprojected_to_utm = True
                print(f"Successfully reprojected to UTM zone: {crs_for_area_calculation.name} (EPSG:{crs_for_area_calculation.to_epsg() if crs_for_area_calculation.to_epsg() else 'N/A'})")
            else:
                print("Could not automatically determine a suitable UTM CRS. Area will be calculated in original geographic units (square degrees).")
        except Exception as e:
            print(f"Error during reprojection to UTM: {e}")
            print("Area will be calculated in original geographic units (square degrees).")
    elif original_crs and original_crs.is_projected:
        #print(f"Original CRS ({original_crs.name}) is already projected. Area will be calculated in its native square units.")
        crs_for_area_calculation = original_crs # Already set, but for clarity
    elif not original_crs:
        print("Warning: CRS is not defined. Area calculation units are unknown and potentially meaningless.")
        # crs_for_area_calculation is already None or original_crs (which is None)

    # Filter for Polygon and MultiPolygon geometries
    polygons_gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    if polygons_gdf.empty:
        print("No Polygon or MultiPolygon features found in the shapefile.")
        return 0.0, crs_for_area_calculation, reprojected_to_utm

    try:
        # The .area attribute directly gives the area in the units of the current CRS
        individual_areas = polygons_gdf.geometry.area
        total_area = individual_areas.sum()
    except Exception as e:
        print(f"Error calculating area: {e}")
        print("This can sometimes happen if geometries are invalid.")
        return None, crs_for_area_calculation, reprojected_to_utm

    return total_area, crs_for_area_calculation, reprojected_to_utm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the combined area of polygons in a shapefile. "
                    "If input CRS is geographic, attempts to convert to UTM for area in square meters."
    )
    parser.add_argument("shapefile_path", type=str, help="Path to the .shp file")
    parser.add_argument("-dp", "--decimal_places", type=int, default=2, help="Number of decimal places for area output (default: 2)")


    args = parser.parse_args()

    #print(f"Processing shapefile: {args.shapefile_path}")
    combined_area, final_crs_obj, was_reprojected = get_combined_polygon_area(args.shapefile_path)

    if combined_area is not None:
        #print(f"CRS Used for Area Calculation: {final_crs_obj.to_wkt(pretty=True) if final_crs_obj else 'Unknown'}")

        unit_name = "units"
        area_format = f".{args.decimal_places}f"

        if final_crs_obj:
            if final_crs_obj.is_projected:
                # Try to get the linear unit name (e.g., "metre", "foot")
                try:
                    unit_name = final_crs_obj.axis_info[0].unit_name.lower()
                    if unit_name == "metre": # Standardize for common spelling
                        unit_name = "meter"
                except (IndexError, AttributeError):
                    unit_name = "units (of projection)" # Fallback
                print(f"{combined_area:.{args.decimal_places}f}")
            elif final_crs_obj.is_geographic:
                unit_name = "degrees"
                print(f"{combined_area:.{args.decimal_places}f}")
                print("Warning: Area calculated in square degrees. For meaningful surface area, reproject to a suitable projected CRS.")
            else: # Neither projected nor geographic (should be rare with valid CRS)
                 print(f"{combined_area:.{args.decimal_places}f}")
        else: # No CRS info
            print(f"{combined_area:.{args.decimal_places}f}")


    else:
        print("Could not calculate the combined area.")