import argparse
import rasterio
import numpy as np # For array operations
import sys         # For printing errors to stderr

def calculate_masked_raster_area(raster_path, target_value=255):
    """
    Calculates the geographic area covered by a raster, considering only pixels
    where the last band equals the target_value.

    Args:
        raster_path (str): Path to the raster file (e.g., GeoTIFF orthomosaic).
        target_value (int): Value in the last band to identify pixels for area calculation.

    Returns:
        tuple: (area_in_native_square_units, native_units_name_str, is_crs_meter_based)
               Returns (None, None, False) if an error occurs.
               native_units_name_str will be 'metres', 'feet', 'degrees', or 'undefined units'.
               is_crs_meter_based is True if linear units are meters.
    """
    try:
        with rasterio.open(raster_path) as src:
            if src.count == 0:
                print(f"Error: Raster file '{raster_path}' has no bands.", file=sys.stderr)
                return None, None, False

            # Read the last band (band indexing in rasterio.read is 1-based)
            last_band_data = src.read(src.count)
            
            # Create a boolean mask where the last band equals the target_value
            mask = (last_band_data == target_value)
            
            # Count the number of True pixels (pixels to include in area calculation)
            valid_pixel_count = np.sum(mask)
            
            if valid_pixel_count == 0:
                # This is a valid scenario, area will be 0.
                # print(f"Info: No pixels with value {target_value} found in the last band of '{raster_path}'. Area is 0.", file=sys.stderr)
                pass

            # Resolution (pixel_width, pixel_height) in CRS units
            # abs() is used because y_resolution (src.res[1]) is often negative
            pixel_area_native_sq_units = abs(src.res[0] * src.res[1])
            
            total_area_native_sq_units = valid_pixel_count * pixel_area_native_sq_units

            # Determine CRS units for context and warnings
            native_units_name = "undefined units"
            is_meter_based = False
            if src.crs:
                if src.crs.is_projected:
                    linear_units = str(src.crs.linear_units).lower() if src.crs.linear_units else ""
                    if 'metre' in linear_units or 'meter' in linear_units:
                        native_units_name = "meters" # Using 'meters' for consistency in output
                        is_meter_based = True
                    elif linear_units:
                        native_units_name = src.crs.linear_units
                    else:
                        native_units_name = "projected units (unspecified)"
                elif src.crs.is_geographic:
                    native_units_name = "degrees"
            
            return total_area_native_sq_units, native_units_name, is_meter_based
                
    except Exception as e:
        print(f"Error processing raster file '{raster_path}': {e}", file=sys.stderr)
        return None, None, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the area of an orthomosaic based on pixels where the last band has a specific value."
    )
    parser.add_argument(
        "--ortho", 
        required=True, 
        help="Path to the orthomosaic GeoTIFF file."
    )
    parser.add_argument(
        "--mask_value",
        type=int,
        default=255,
        help="Value in the last band to identify pixels to include in area calculation. Default: 255"
    )
    parser.add_argument(
        "--unit", 
        default="m2", 
        choices=["m2", "ha", "acres", "native"], 
        help="Output unit for area (m2, ha, acres, or native square units of the CRS). Default: m2"
    )
    parser.add_argument(
        "--decimals",
        default=2,
        type=int,
        help="Number of decimal places for the output area. Default: 2"
    )
    
    args = parser.parse_args()

    area_native_units, native_unit_name, is_meter_based = calculate_masked_raster_area(args.ortho, args.mask_value)

    if area_native_units is not None:
        output_area = area_native_units
        
        if args.unit == "native":
            pass # Outputting in native square units
        elif is_meter_based: # CRS units are meters, so conversions are valid
            if args.unit == "ha":
                output_area = area_native_units / 10000.0
            elif args.unit == "acres":
                output_area = area_native_units / 4046.8564224 # 1 acre = 4046.8564224 m^2
            elif args.unit == "m2":
                output_area = area_native_units # Already in sq meters
        else: # CRS units are not meters (e.g., degrees, feet) or undefined
            if args.unit != "native": # User requested conversion but base units are not meters
                print(f"Warning: Input raster CRS is not in meters (units: {native_unit_name}). "
                      f"Conversion to {args.unit} from non-meter base units may be inaccurate. "
                      f"Outputting in native square {native_unit_name} instead. Use --unit native or reproject input.",
                      file=sys.stderr)
            # In this case (non-meter base and conversion requested), we still output native value.
            # The shell script needs to be aware of this possibility if strict units are required.
            # Alternatively, we could make the script error out here if conversion is not possible.
            # For now, we output the native value if conversion is problematic.
            pass # output_area remains area_native_units

        print(f"{output_area:.{args.decimals}f}")
    else:
        print("Error:AreaCalculationFailed", file=sys.stdout) # For shell script to check
        sys.exit(1) # Exit with an error code