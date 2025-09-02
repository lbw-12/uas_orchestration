import geopandas as gpd
from shapely.geometry import Point, shape, mapping, box, Polygon
import rasterio.mask
from rasterio.features import rasterize
import os
import csv
import rasterio
import time
import argparse
import shutil
import cv2
import numpy as np
from scipy.ndimage import rotate
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
import math
import matplotlib.pyplot as plt
import pdb
from rasterio.transform import rowcol
from math import atan2, degrees
from collections import defaultdict
import re
from rasterio.mask import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import affine
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform
import math
import concurrent.futures
from pathlib import Path
import pandas as pd


def calculate_distance(point1, point2):
    return Point(point1).distance(Point(point2))

def intersects(feature, raster_bounds):
    feature_geom = shape(feature["geometry"])
    raster_geom = box(*raster_bounds)
    return raster_geom.contains(feature_geom)

def add_custom_metadata(input_path, output_path, metadata):
    with rasterio.open(input_path, 'r+') as dataset:
        dataset.update_tags(**metadata)

def plot_shape_on_masked_image(out_image, shape_coordinates, out_transform):
    if len(out_image.shape) == 3:
        image_to_plot = out_image[0]
    else:
        image_to_plot = out_image
    def geo_to_pixel(x, y, transform):
        col, row = ~transform * (x, y)
        return int(col), int(row)

    pixel_coords = [geo_to_pixel(x, y, out_transform) for x, y in shape_coordinates]
    x_coords, y_coords = zip(*pixel_coords)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_to_plot, cmap='gray')
    plt.savefig("plots/image_to_plot.png")
    plt.plot(x_coords, y_coords, color='red', linewidth=2, label="Polygon Shape")
    plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Vertices')
    
    plt.legend()
    plt.title("Polygon Shape on Cropped Image")
    plt.savefig("plots/polygon_shape_on_image.png")

def crop_black_white_pixels(rotated_bands, plot_no, date, location, sensor, threshold=0.0, border=5):
    x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0
    has_valid_band = False

    for band in rotated_bands:
        nonzero_indices = np.where(band > threshold)
        if nonzero_indices[0].size > 0:
            has_valid_band = True
            x_min = min(x_min, np.min(nonzero_indices[1]))
            x_max = max(x_max, np.max(nonzero_indices[1]))
            y_min = min(y_min, np.min(nonzero_indices[0]))
            y_max = max(y_max, np.max(nonzero_indices[0]))

    if not has_valid_band:
        #cropped_bands = np.array([])
        print(f"No valid non-zero pixels found for plot {plot_no}. Check image and threshold.")
        return [], has_valid_band

    x_min, x_max = max(x_min + border, 0), max(x_max - border, 0)
    y_min, y_max = max(y_min + border, 0), max(y_max - border, 0)

    cropped_bands = [band[y_min:y_max+1, x_min:x_max+1] for band in rotated_bands]
    h, w = cropped_bands[0].shape

    if h > w:
        cropped_bands = [cv2.rotate(band, cv2.ROTATE_90_CLOCKWISE) for band in cropped_bands]
        print(f"Rotated vertically aligned image for plot {plot_no} to horizontal")

    """
    for i, cropped_band in enumerate(cropped_bands):
        plt.figure()
        plt.imshow(cropped_band, cmap='gray')
        plt.title(f"Cropped Image - Band {i+1}")
        plt.colorbar()
        plt.savefig(f"plots/cropped_band_{i+1}_{date}_{location}_{plot_no}_{sensor}.png")
        plt.close()
        break
    """
    return cropped_bands, has_valid_band

def get_longest_edge_angle(coordinates, out_transform):
    points = np.array(coordinates[0])

    def geo_to_pixel(x, y, transform):
        col, row = ~transform * (x, y)
        return [int(col), int(row)]
    pixel_coords = [geo_to_pixel(x, y, out_transform) for x, y in points]
    pixel_coords = np.array(pixel_coords[:-1]) 
    centroid = np.mean(pixel_coords, axis=0)

    def angle_from_centroid(pixel_coords):
        return np.arctan2(pixel_coords[1] - centroid[1], pixel_coords[0] - centroid[0])
    pixel_coords = sorted(pixel_coords, key=angle_from_centroid)
    edge_lengths = [np.linalg.norm(pixel_coords[i] - pixel_coords[(i + 1) % 4]) for i in range(4)]
    width = (edge_lengths[0] + edge_lengths[2]) / 2
    height = (edge_lengths[1] + edge_lengths[3]) / 2

    try:
        slope = (pixel_coords[1][1] - pixel_coords[0][1]) / (pixel_coords[1][0] - pixel_coords[0][0])
    except ZeroDivisionError:
        slope = float('inf')
    angle = math.atan(slope) * 180 / math.pi if slope != float('inf') else 90.0
    try:
        slope_vert = (pixel_coords[3][1] - pixel_coords[0][1]) / (pixel_coords[3][0] - pixel_coords[0][0])
    except ZeroDivisionError:
        slope_vert = float('inf')
    angle_vert = math.atan(slope_vert) * 180 / math.pi if slope_vert != float('inf') else 90.0
    if angle_vert < 0:
        angle2 = 90 + angle_vert
    else:
        angle2 = angle_vert
    return angle2 if height > width else angle

def plot_shape_on_image(src, shape_coords):

    img = src.read(1)  
    bounds = src.bounds
    left, bottom, right, top = bounds
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[left, right, bottom, top], cmap='gray')
    x_coords, y_coords = zip(*shape_coords)
    plt.plot(x_coords, y_coords, color='blue', linewidth=2, marker='o', markersize=5, label='Shape Coordinates')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Shape on Full Image')
    plt.legend()
    plt.grid(False)
    plt.savefig("plots/shape_on_image.png")

def _get_longest_edge_angle_from_geom(geom):
    """
    Calculates the normalized angle of a polygon's primary orientation
    by ensuring the final angle is always between -90 and +90 degrees.
    """
    mrr = geom.minimum_rotated_rectangle
    points = list(mrr.exterior.coords)

    if not points:
        return 0.0

    max_len_sq = 0
    longest_segment = (0, 0, 0, 0)

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        len_sq = (x2 - x1)**2 + (y2 - y1)**2
        if len_sq > max_len_sq:
            max_len_sq = len_sq
            longest_segment = (x1, y1, x2, y2)

    p1x, p1y, p2x, p2y = longest_segment
    angle_rad = math.atan2(p2y - p1y, p2x - p1x)
    angle_deg = math.degrees(angle_rad)

    # Normalize the angle to be between -90 and 90 degrees.
    # This corrects for 180-degree flips in direction.
    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0

    return angle_deg

def get_plot_tile_geospatial(row, image_path, plot_no, date, location, sensor):
    """
    Corrected hybrid version. Uses rasterio for a simple clip and scipy
    for a precise rotation, with robust math for the final georeferencing.
    """
    with rasterio.open(image_path) as src:
        global crop_black_white_pixels

        # === PART 1: Simple Geospatial Clip ===
        # First, clip a simple, north-up bounding box around the plot.
        buffered_geometry = row["geometry"].buffer(5 * abs(src.transform.a))
        try:
            clipped_image, clipped_transform = rasterio.mask.mask(
                src, [mapping(buffered_geometry)], crop=True, filled=True, nodata=0
            )
        except ValueError:
            print(f"Warning: Plot {plot_no} appears to be outside the raster bounds.")
            return plot_no, None, False, None

        # === PART 2: Precise Image Rotation with SciPy ===
        # Get the angle and rotate the image array. The negative angle straightens the image.
        angle = _get_longest_edge_angle_from_geom(row["geometry"])
        
        rotated_bands = []
        for i in range(clipped_image.shape[0]):
            rotated_band = rotate(
                clipped_image[i], -angle, reshape=True, mode='constant', cval=0, order=1
            )
            rotated_bands.append(rotated_band)
        rotated_image = np.array(rotated_bands, dtype=clipped_image.dtype)

        # === PART 3: Calculate the New GeoTransform (Corrected) ===
        # This robustly calculates the final georeferenced position.
        
        # Get the geographic coordinate of the image's center point
        center_pixel = (clipped_image.shape[2] / 2.0, clipped_image.shape[1] / 2.0)
        center_geo = clipped_transform * center_pixel
        
        # Get the pixel coordinate of the new rotated image's center
        new_center_pixel = (rotated_image.shape[2] / 2.0, rotated_image.shape[1] / 2.0)
        
        # Create the rotation/scale part of the new transform
        pixel_size = src.res[0]
        rotation_and_scale = affine.Affine.rotation(-angle) * affine.Affine.scale(pixel_size, -pixel_size)

        # Calculate the translation part (the top-left corner's coordinate)
        rotated_center_offset = rotation_and_scale * new_center_pixel
        c = center_geo[0] - rotated_center_offset[0]
        f = center_geo[1] - rotated_center_offset[1]
        
        # Combine the translation with the rotation and scale
        final_transform = affine.Affine(
            rotation_and_scale.a, rotation_and_scale.b, c,
            rotation_and_scale.d, rotation_and_scale.e, f
        )
        
        # === PART 4: Final Cleanup ===
        # Use your existing function to crop away the black borders.
        cropped_image, has_valid_band = crop_black_white_pixels(
            list(rotated_image), plot_no, date, location, sensor
        )
        
        final_cropped = np.array(cropped_image, dtype=src.dtypes[0])
        return plot_no, final_cropped, has_valid_band, final_transform

def get_plot_tile_geospatial_old2(row, image_path, plot_no, date, location, sensor):
    """
    Fixed version that properly handles coordinate transformations.
    Key fix: Use the rotated geometry for masking in the rotated coordinate system.
    """
    
    with rasterio.open(image_path) as src:
        global crop_black_white_pixels

        # Use the buffered geometry for all calculations
        pixel_size = abs(src.transform.a)  # Get pixel size once
        buffered_geometry = row["geometry"].buffer(5 * pixel_size)

        # 1. Get angle from the buffered geometry
        angle = _get_longest_edge_angle_from_geom(buffered_geometry)

        # 2. Get the centroid of the buffered geometry as pivot point
        bounds = buffered_geometry.bounds
        pivot_x = (bounds[0] + bounds[2]) / 2
        pivot_y = (bounds[1] + bounds[3]) / 2
        pivot_coords = (pivot_x, pivot_y)

        # 3. Create rotation transformation around the centroid
        rot_transform = affine.Affine.rotation(angle, pivot=pivot_coords)
        
        # Apply rotation to get rotated geometry
        rotated_geom = shapely_transform(
            lambda x, y, z=None: rot_transform * (x, y), 
            buffered_geometry
        )
        
        # 4. Calculate the bounding box of the rotated geometry
        rot_bounds = rotated_geom.bounds
        
        # 5. Add buffer in pixels
        buffer_px = 50
        buffer_size = buffer_px * pixel_size
        
        # Expand bounds by buffer
        expanded_bounds = (
            rot_bounds[0] - buffer_size,  # min_x
            rot_bounds[1] - buffer_size,  # min_y
            rot_bounds[2] + buffer_size,  # max_x
            rot_bounds[3] + buffer_size   # max_y
        )
        
        # 6. Calculate output dimensions
        width_m = expanded_bounds[2] - expanded_bounds[0]
        height_m = expanded_bounds[3] - expanded_bounds[1]
        
        dst_width = int(math.ceil(width_m / pixel_size))
        dst_height = int(math.ceil(height_m / pixel_size))
        
        # Ensure dimensions are positive and reasonable
        if dst_width <= 0 or dst_height <= 0:
            print(f"Warning: Plot {plot_no} has invalid dimensions: {dst_width}x{dst_height}")
            return plot_no, None, False, None
            
        if dst_width > 10000 or dst_height > 10000:
            print(f"Warning: Plot {plot_no} has very large dimensions: {dst_width}x{dst_height}")
            return plot_no, None, False, None

        # 7. Create destination transform for the rotated coordinate system
        top_left_x = expanded_bounds[0]
        top_left_y = expanded_bounds[3]  # Top in image coordinates
        
        # Create the destination transform
        dst_transform = affine.Affine(
            pixel_size, 0.0, top_left_x,
            0.0, -pixel_size, top_left_y
        )

        # --- DEBUGGING BLOCK ---
        debug_dir = "debug_shapefiles_om"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create a polygon representing the VRT canvas
        vrt_corner1 = dst_transform * (0, 0)
        vrt_corner2 = dst_transform * (dst_width, dst_height)
        vrt_canvas_geom = box(
            min(vrt_corner1[0], vrt_corner2[0]),
            min(vrt_corner1[1], vrt_corner2[1]),
            max(vrt_corner1[0], vrt_corner2[0]),
            max(vrt_corner1[1], vrt_corner2[1])
        )
        
        # Save debug shapefiles
        try:
            gpd.GeoDataFrame(
                {'id': [plot_no], 'type': ['buffered']}, 
                geometry=[buffered_geometry], 
                crs=src.crs
            ).to_file(os.path.join(debug_dir, f"plot_{plot_no}_1_buffered.shp"))
            
            gpd.GeoDataFrame(
                {'id': [plot_no], 'type': ['rotated']}, 
                geometry=[rotated_geom], 
                crs=src.crs
            ).to_file(os.path.join(debug_dir, f"plot_{plot_no}_2_rotated.shp"))
            
            gpd.GeoDataFrame(
                {'id': [plot_no], 'type': ['vrt_canvas']}, 
                geometry=[vrt_canvas_geom], 
                crs=src.crs
            ).to_file(os.path.join(debug_dir, f"plot_{plot_no}_3_vrt_canvas.shp"))
        except Exception as e:
            print(f"Debug shapefile creation failed for plot {plot_no}: {e}")

        # 8. Create the WarpedVRT - KEY FIX: Apply the rotation to the VRT itself
        try:
            # Create the inverse rotation transform to apply to the VRT
            # This rotates the imagery so that when we mask with the rotated geometry,
            # everything aligns correctly
            inv_rot_transform = affine.Affine.rotation(-angle, pivot=pivot_coords)
            
            # Combine the source transform with the inverse rotation
            vrt_transform = src.transform * inv_rot_transform
            
            with WarpedVRT(src,
                        crs=src.crs,
                        transform=vrt_transform,
                        width=dst_width,
                        height=dst_height,
                        resampling=Resampling.bilinear,
                        nodata=src.nodata) as vrt:

                # 9. CRITICAL FIX: Use the rotated geometry for masking
                # Now the VRT is rotated, so we use the rotated geometry
                
                # Check if the rotated geometry overlaps with the VRT bounds
                vrt_bounds = vrt.bounds
                geom_bounds = rotated_geom.bounds
                
                # Quick overlap check
                if (geom_bounds[2] < vrt_bounds[0] or  # geom max_x < vrt min_x
                    geom_bounds[0] > vrt_bounds[2] or  # geom min_x > vrt max_x
                    geom_bounds[3] < vrt_bounds[1] or  # geom max_y < vrt min_y
                    geom_bounds[1] > vrt_bounds[3]):   # geom min_y > vrt max_y
                    print(f"Warning: Plot {plot_no} rotated geometry doesn't overlap VRT bounds")
                    print(f"  Rotated geometry bounds: {geom_bounds}")
                    print(f"  VRT bounds: {vrt_bounds}")
                    return plot_no, None, False, None

                # Use the rotated geometry for masking
                shapes = [mapping(rotated_geom)]
                
                try:
                    # Apply masking with the rotated geometry
                    rotated_image, out_transform = rasterio.mask.mask(
                        vrt,
                        shapes,
                        crop=True,
                        all_touched=True,
                        filled=True
                    )
                    
                except ValueError as e:
                    print(f"Warning: Plot {plot_no} could not be masked. Error: {e}")
                    
                    # Try alternative approach: manually extract the region
                    try:
                        # Get the window that covers the rotated geometry
                        window = rasterio.windows.from_bounds(
                            *rotated_geom.bounds, 
                            transform=vrt.transform
                        )
                        
                        # Read the data from the window
                        rotated_image = vrt.read(window=window)
                        out_transform = rasterio.windows.transform(window, vrt.transform)
                        
                        print(f"Plot {plot_no}: Used window-based extraction as fallback")
                        
                    except Exception as e2:
                        print(f"Warning: Plot {plot_no} fallback extraction failed: {e2}")
                        return plot_no, None, False, None
                    
                # Check if we got valid data
                if rotated_image is None or rotated_image.size == 0:
                    print(f"Warning: Plot {plot_no} produced empty image after masking")
                    return plot_no, None, False, None

        except Exception as e:
            print(f"Error creating WarpedVRT for plot {plot_no}: {e}")
            return plot_no, None, False, None

        # 10. Post-process
        try:
            cropped_image, has_valid_band = crop_black_white_pixels(
                list(rotated_image), plot_no, date, location, sensor
            )
            
            # 11. Convert to final numpy array
            if cropped_image is not None:
                final_cropped = np.array(cropped_image, dtype=src.dtypes[0])
                return plot_no, final_cropped, has_valid_band, out_transform
            else:
                return plot_no, None, False, None
                
        except Exception as e:
            print(f"Error in post-processing for plot {plot_no}: {e}")
            return plot_no, None, False, None

# This function is a drop-in replacement for the original get_plot_tile function.
def get_plot_tile_geospatial_old(row, image_path, plot_no, date, location, sensor):
    """
    The version that was producing a result with only a small (5cm) offset.
    Uses the bounding box center as the pivot point.
    """
    with rasterio.open(image_path) as src:
        global crop_black_white_pixels

        # Use the buffered geometry for all calculations for consistency
        buffered_geometry = row["geometry"].buffer(5 * abs(src.transform.a))

        # 1. Get angle from the buffered geometry
        angle = _get_longest_edge_angle_from_geom(buffered_geometry)

        print(f'angle: {angle}')

        # 2. Define rotation using the BOUNDING BOX CENTER as the pivot
        bounds = buffered_geometry.bounds
        pivot_coords = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

        rot_around_center = affine.Affine.rotation(angle, pivot=pivot_coords)
        rotated_geom = shapely_transform(lambda x, y, z=None: rot_around_center * (x, y), buffered_geometry)
        dst_bounds = rotated_geom.bounds

        # 3. Build the destination transform
        buffer_px = 50
        pixel_size = src.res[0]
        top_left_x = dst_bounds[0] - (buffer_px * pixel_size)
        top_left_y = dst_bounds[3] + (buffer_px * pixel_size)

        dst_transform = (
            affine.Affine.translation(top_left_x, top_left_y) *
            affine.Affine.rotation(angle) *
            affine.Affine.scale(pixel_size, -pixel_size)
        )

        # 4. Calculate output dimensions
        dst_width = math.ceil(abs(dst_bounds[2] - dst_bounds[0]) / pixel_size) + (buffer_px * 2)
        dst_height = math.ceil(abs(dst_bounds[3] - dst_bounds[1]) / pixel_size) + (buffer_px * 2)

        # --- START DEBUGGING BLOCK ---
        debug_dir = "debug_shapefiles_om"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create a polygon representing the VRT canvas
        vrt_b = dst_transform * (0, 0)
        vrt_c = dst_transform * (dst_width, dst_height)
        vrt_canvas_geom = box(vrt_b[0], vrt_c[1], vrt_c[0], vrt_b[1])
        
        # Save the two key shapes to uniquely named files
        gpd.GeoDataFrame({'id': [plot_no]}, geometry=[buffered_geometry], crs=src.crs).to_file(os.path.join(debug_dir, f"plot_{plot_no}_1_buffered.shp"))
        gpd.GeoDataFrame({'id': [plot_no]}, geometry=[vrt_canvas_geom], crs=src.crs).to_file(os.path.join(debug_dir, f"plot_{plot_no}_2_vrt_canvas.shp"))
        # --- END DEBUGGING BLOCK ---

        # 5. Create the WarpedVRT
        with WarpedVRT(src,
                    crs=src.crs,
                    transform=dst_transform,
                    width=dst_width,
                    height=dst_height,
                    resampling=Resampling.bilinear,
                    nodata=src.nodata) as vrt:

            # 6. Mask from the VRT using the same buffered geometry
            shapes = [mapping(buffered_geometry)]
            try:
                rotated_image, out_transform = rasterio.mask.mask(
                    vrt,
                    shapes,
                    crop=True,
                    all_touched=True,
                    filled=True
                )
            except ValueError as e:
                print(f"Warning: Plot {plot_no} could not be masked. Error: {e}")
                return None, False, None

        # 7. Post-process
        cropped_image, has_valid_band = crop_black_white_pixels(
            list(rotated_image), plot_no, date, location, sensor
        )
        
        # 8. Convert to final numpy array
        final_cropped = np.array(cropped_image, dtype=src.dtypes[0])
        return plot_no,final_cropped, has_valid_band, out_transform

def extract_plot_tiles_om(shapefile_path, output_path, image_file, crop, location, date):
    shpfile = gpd.read_file(shapefile_path)
    
    processed_plots = set()
    processed_plots_info = {}
    output_filenames = {}
    useful_files = set()
    
    count = 0
    
    raster_lower = image_file.lower()
    if "rgb" in raster_lower:
        sensor = 'rgb'
        #output_specific_path = str(output_path) + "plot_tiles_rgb_om/" 
    elif "ms" in raster_lower:
        sensor = 'ms'
        #output_specific_path = str(output_path) + "plot_tiles_ms_om/"
    
    with rasterio.open(image_file) as src:
        print(f'shapefile path: {shapefile_path}')
        print(f'shpfile crs to epsg: {shpfile.crs.to_epsg()}')
        print(f'image path: {image_file}')
        print(f'src.crs to epsg: {src.crs.to_epsg()}')
        print(f'src.bounds: {src.bounds}')
        print(f'src.transform: {src.transform}')
        if shpfile.crs.to_epsg() != src.crs.to_epsg():
            raise ValueError("Shapefile and OM CRS do not match.", shapefile_path, image_file)
        raster_center = [(src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2]
        print(f'Length of shapefile: {len(shpfile)}')
        
        for index, row in shpfile.sort_values(by='plot').iterrows():
            print(f'row: {row}')
            if "plot" not in row:
                print(f'Plot not found in shapefile: {row}')
                continue
            plot_no = row["plot"]
            plot_center = row["geometry"].centroid.coords[0]
            distance_to_image_center = calculate_distance(plot_center, raster_center)
            if plot_no in processed_plots_info:
                continue
            if intersects(row, src.bounds):
                print(f'Intersects')
                #final_cropped, has_valid_band, out_transform = get_plot_tile(row, src, plot_no, date, location, sensor)
                _, final_cropped, has_valid_band, out_transform = get_plot_tile_geospatial(row, image_file, plot_no, date, location, sensor)
                out_shape = final_cropped.shape[1:]
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_shape[0],
                    "width": out_shape[1],
                    "transform": out_transform
                })
                out_meta["height"] = final_cropped.shape[1]
                out_meta["width"] = final_cropped.shape[2]
                output_filename = os.path.join(output_path, f"{location}_om_{crop}_{sensor}_{plot_no}_{date}.tif")
                print(output_filename)
                with rasterio.open(output_filename, "w", **out_meta) as dest:
                    dest.write(final_cropped)
                metadata = {
                    'DATE': date,
                    'LOCATION': location,
                    'PLOT NO': str(plot_no),
                    'SENSOR': sensor
                }
                
                add_custom_metadata(output_filename, output_filename, metadata)
                processed_plots.add(plot_no)
                processed_plots_info[plot_no] = {
                    "distance": distance_to_image_center,
                    "raster_file": output_filename
                }

                count += 1
                output_filenames[plot_no] = {"raster_basename": sensor}

    return output_filenames

def extract_plot_tiles_om_parallel(shapefile_path, output_path, image_file, crop, location, date):
    shpfile = gpd.read_file(shapefile_path)
    os.makedirs(output_path, exist_ok=True) # Ensure output directory exists
    
    output_filenames = {}

    # Determine sensor type from filename
    sensor = 'rgb' if "rgb" in image_file.lower() else 'ms'

    # Use ProcessPoolExecutor to manage parallel processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        
        # --- SUBMIT JOBS ---
        # Create a dictionary to map each running job (future) to its plot number
        futures = {
            executor.submit(get_plot_tile_geospatial, row, image_file, row['plot'], date, location, sensor): row['plot']
            for index, row in shpfile.sort_values(by='plot').iterrows()
        }

        # --- PROCESS RESULTS ---
        # Loop over the jobs as they complete
        for future in concurrent.futures.as_completed(futures):
            plot_no = futures[future]
            try:
                # Get the result from the completed job
                returned_plot_no, final_cropped, has_valid_band, out_transform = future.result()

                # If the worker function returned a valid image, save it
                if has_valid_band and final_cropped is not None and final_cropped.size > 0:
                    out_meta = {
                        "driver": "GTiff",
                        "height": final_cropped.shape[1],
                        "width": final_cropped.shape[2],
                        "count": final_cropped.shape[0],
                        "dtype": final_cropped.dtype,
                        "crs": shpfile.crs, # Use CRS from the shapefile
                        "transform": out_transform,
                        "nodata": 0 # Or your specific nodata value
                    }
                    output_filename = os.path.join(output_path, f"{location}_om_{crop}_{sensor}_{plot_no}_{date}.tif")
                    
                    with rasterio.open(output_filename, "w", **out_meta) as dest:
                        dest.write(final_cropped)
                    
                    print(f"Successfully created: {os.path.basename(output_filename)}")

                    # Add your custom metadata
                    metadata = {'DATE': date, 'LOCATION': location, 'PLOT NO': str(plot_no), 'SENSOR': sensor}
                    # add_custom_metadata(output_filename, output_filename, metadata) # Assuming you have this function
                    
                    output_filenames[plot_no] = {"raster_basename": sensor}

            except Exception as exc:
                print(f'Plot {plot_no} generated an exception during processing: {exc}')

    return output_filenames

def extract_plot_tiles_ir(shapefile_path, output_path, sorted_raster_files, crop, location, date):
    shpfile = gpd.read_file(shapefile_path)
    processed_plots = set()
    processed_plots_info = {}
    output_filenames = {}
    useful_files = set()
    plot_candidates = defaultdict(list)
    count = 0
    for raster_file in sorted_raster_files:
        raster_lower = raster_file.lower()
        
        if "sony" in raster_lower:
            sensor = 'rgb'
            if "ir" in raster_lower:
                output_specific_path = str(output_path) + "plot_tiles_rgb_ir/" 
            elif "dgr" in raster_lower:
                output_specific_path = str(output_path) + "plot_tiles_rgb_dgr/"
        elif "altum" in raster_lower:
            sensor = 'ms'
            if "ir" in raster_lower:
                output_specific_path = str(output_path) + "plot_tiles_ms_ir/"
            elif "dgr" in raster_lower:
                output_specific_path = str(output_path) + "plot_tiles_ms_dgr/"
        
        with rasterio.open(raster_file) as src:
            
            all_bands = src.read()
            non_zero_mask = np.any(all_bands != 0, axis=0)
            alpha_band_custom = non_zero_mask.astype(np.uint8) * 255
            if shpfile.crs != src.crs:
                raise ValueError("Shapefile and IR CRS do not match.", shapefile_path, raster_file)
            raster_center = [(src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2]
            for index, row in shpfile.sort_values(by='plot').iterrows():
                if "plot" not in row:
                    continue

                if not intersects(row, src.bounds):
                    continue
                plot_no = row["plot"]
                plot_geom = mapping(row["geometry"])
                plot_mask = geometry_mask(
                    geometries=[plot_geom],
                    transform=src.transform,
                    invert=True,
                    out_shape=(src.height, src.width)
                )

                alpha_within_plot = alpha_band_custom[plot_mask]

                if np.all(alpha_within_plot == 255):
                    distance_to_image_center = calculate_distance(row["geometry"].centroid.coords[0], raster_center)
                    plot_candidates[plot_no].append((distance_to_image_center, raster_file, sensor, output_specific_path))
            """
            raster_center = [(src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2]
            
            for index, row in shpfile.iterrows():
                
                if "plot" not in row:
                    continue
                plot_no = row["plot"]
                plot_center = row["geometry"].centroid.coords[0]
                distance_to_image_center = calculate_distance(plot_center, raster_center)
                if intersects(row, src.bounds):
                    plot_candidates[plot_no].append((distance_to_image_center, raster_file, sensor, output_specific_path))
            """
    if sensor == "ms":
        for plot_no, candidates in plot_candidates.items():
            grouped_candidates = defaultdict(list)
        
            for dist, filename, sensor, output_specific_path in candidates:
                match = re.search(r"_(\d+)\.tif$", filename)
                base_name = match.group(1)
                grouped_candidates[base_name].append((dist, filename, sensor, output_specific_path))
            best_per_group = {}
            for base_name, versions in grouped_candidates.items():
                best = min(versions, key=lambda x: x[0])  
                best_per_group[base_name]=best
        
            for base_name in best_per_group.keys():
                distance, raster_file, sensor, output_specific_path = best_per_group[base_name]
                with rasterio.open(raster_file) as src:
                    row = shpfile[shpfile["plot"] == plot_no].iloc[0]
                    #final_cropped, has_valid_band, out_transform = get_plot_tile(row, src, plot_no, date, location, sensor)
                    _, final_cropped, has_valid_band, out_transform = get_plot_tile_geospatial(row, raster_file, plot_no, date, location, sensor)
                    if has_valid_band:
                        out_shape = final_cropped.shape[1:]
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_shape[0],
                            "width": out_shape[1],
                            "transform": out_transform
                        })
                        output_filename = os.path.join(output_specific_path, f"{location}_ir_{crop}_{sensor}_{plot_no}_{base_name}_{date}.tif")
                        print(output_filename)
                        with rasterio.open(output_filename, "w", **out_meta) as dest:
                            dest.write(final_cropped)
                        metadata = {
                                'DATE': date,
                                'LOCATION': location,
                                'PLOT NO': str(plot_no),
                                'SENSOR': sensor
                            }   
                        add_custom_metadata(output_filename, output_filename, metadata)
                        processed_plots.add(plot_no)
                        processed_plots_info[plot_no] = {
                            "distance": distance_to_image_center,
                            "raster_file": output_filename
                        }
                        count += 1
                        output_filenames[plot_no] = {"raster_basename": sensor}
    elif sensor == "rgb":
        
        for plot_no, candidates in plot_candidates.items():
            best = min(candidates, key=lambda x: x[0])
            distance, raster_file, sensor, output_specific_path = best
            with rasterio.open(raster_file) as src:
                row = shpfile[shpfile["plot"] == plot_no].iloc[0]
                #final_cropped, has_valid_band, out_transform = get_plot_tile(row, src, plot_no, date, location, sensor)
                _, final_cropped, has_valid_band, out_transform = get_plot_tile_geospatial(row, raster_file, plot_no, date, location, sensor)
                if has_valid_band:
                        
                    out_shape = final_cropped.shape[1:]
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_shape[0],
                        "width": out_shape[1],
                        "transform": out_transform
                    })
                    output_filename = os.path.join(output_specific_path, f"{location}_ir_{crop}_{sensor}_{plot_no}_{date}.tif")
                    print(output_filename)
                    with rasterio.open(output_filename, "w", **out_meta) as dest:
                        dest.write(final_cropped)
                    metadata = {
                            'DATE': date,
                            'LOCATION': location,
                            'PLOT NO': str(plot_no),
                            'SENSOR': sensor
                        }   
                    add_custom_metadata(output_filename, output_filename, metadata)
                    processed_plots.add(plot_no)
                    processed_plots_info[plot_no] = {
                        "distance": distance_to_image_center,
                        "raster_file": output_filename
                    }
                    count += 1
                    output_filenames[plot_no] = {"raster_basename": sensor}
    return output_filenames


def find_plot_candidates_in_raster(raster_file, shpfile_path):
    """
    Worker function to find all plot candidates within a single raster file
    based on simple geometric intersection.
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box, Point

    shpfile = gpd.read_file(shpfile_path)
    candidates = []

    with rasterio.open(raster_file) as src:
        src_bounds_geom = box(*src.bounds)
        raster_center_point = src_bounds_geom.centroid

        for index, row in shpfile.sort_values(by='plot').iterrows():
            plot_geom = row["geometry"]
            if "plot" not in row or not plot_geom.is_valid:
                continue

            # This is the simple intersection check you requested
            if src_bounds_geom.intersects(plot_geom):
                plot_no = row["plot"]
                distance = plot_geom.centroid.distance(raster_center_point)
                candidates.append((plot_no, distance, raster_file))

    return candidates

def extract_plot_tiles_ir_parallel(shapefile_path, output_path, sorted_raster_files, crop, location, date):
    shpfile = gpd.read_file(shapefile_path)
    plot_candidates = defaultdict(list)
    output_filenames = {}

    # === PHASE 1: Find Best Candidates (Parallel) ===
    print("Phase 1: Finding best candidate rasters for each plot (in parallel)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(find_plot_candidates_in_raster, raster, shapefile_path): raster for raster in sorted_raster_files}

        for future in concurrent.futures.as_completed(futures):
            try:
                for plot_no, distance, raster_file in future.result():
                    # Assuming worker returns sensor type, if not, determine it here
                    sensor = 'rgb' if "rgb" in raster_file.lower() else 'ms'
                    plot_candidates[plot_no].append((distance, raster_file, sensor))
            except Exception as exc:
                print(f'A raster file generated an exception: {exc}')

    # Create the final list of jobs to run
    jobs_to_run = []
    for plot_no, candidates in plot_candidates.items():
        if not candidates: continue
        best_distance, best_raster_file, sensor = min(candidates, key=lambda x: x[0])
        row = shpfile[shpfile["plot"] == plot_no].iloc[0]
        jobs_to_run.append((row, plot_no, best_raster_file, sensor))

    # === PHASE 2: Submit and Process Extraction Jobs (Parallel) ===
    print(f"\nPhase 2: Submitting {len(jobs_to_run)} plots for parallel extraction...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(get_plot_tile_geospatial, job_row, job_raster_file, job_plot_no, date, location, job_sensor): job_plot_no
            for job_row, job_plot_no, job_raster_file, job_sensor in jobs_to_run
        }

        # --- THIS IS THE FIX: This block saves the results ---
        for future in concurrent.futures.as_completed(futures):
            plot_no = futures[future]
            try:
                returned_plot_no, final_cropped, has_valid_band, out_transform = future.result()

                if has_valid_band and final_cropped is not None and final_cropped.size > 0:
                    # Find the original "best_raster_file" for this plot to get metadata
                    original_job = next(job for job in jobs_to_run if job[1] == returned_plot_no)
                    best_raster_file = original_job[2]
                    sensor = original_job[3]
                    
                    # Create specific output path
                    output_specific_path = output_path
                    os.makedirs(output_specific_path, exist_ok=True)
                    
                    output_filename = os.path.join(output_specific_path, f"{location}_ir_{crop}_{sensor}_{returned_plot_no}_{date}.tif")

                    with rasterio.open(best_raster_file) as src:
                        out_meta = src.meta.copy()
                    
                    out_meta.update({
                        "driver": "GTiff", "height": final_cropped.shape[1],
                        "width": final_cropped.shape[2], "transform": out_transform, "nodata": 0
                    })
                    
                    with rasterio.open(output_filename, "w", **out_meta) as dest:
                        dest.write(final_cropped)

                    print(f"Successfully processed and saved plot {returned_plot_no}")
                    output_filenames[returned_plot_no] = output_filename

            except Exception as exc:
                print(f"Plot {plot_no} generated an exception during extraction: {exc}")

    return output_filenames

def get_sorted_raster_files(df_csv, image_folder_path):

    # --- Step 1: Scan the image folder ONCE to create a lookup map ---
    # The key is the numeric part (e.g., "100_1"), value is the full file path.
    image_file_map = {}
    for image_filename in os.listdir(image_folder_path):
        if image_filename.lower().endswith(".tif"):
            match = re.search(r"(\d+_\d+)", image_filename)
            if match:
                num_part_image = match.group(1)
                image_file_map[num_part_image] = os.path.join(image_folder_path, image_filename)

    omega_phi_list = []

    for index, row in df_csv.iterrows():
        filename_in_csv = row.iloc[0]
        omega_value = float(row.iloc[4])
        phi_value = float(row.iloc[5])
        kappa_value = float(row.iloc[6])
        match = re.search(r"(\d+_\d+)", filename_in_csv)
        num_part_csv = match.group(1)
            
        # Use the map for a fast lookup instead of a nested loop
        if num_part_csv in image_file_map:
            omega_value = float(row.iloc[4])
            phi_value = float(row.iloc[5])
            kappa_value = float(row.iloc[6])
            full_path = image_file_map[num_part_csv]
            omega_phi_list.append((full_path, omega_value, phi_value, kappa_value))

    sorted_raster_files = [item[0] for item in sorted(omega_phi_list, key=lambda x: abs(x[1]) + abs(x[2]))]

    return sorted_raster_files

def get_crop_from_shapefile_path(shapefile_path):
    crop = None
    crop_list = ["soy", "corn", "cc", "sc"]
    for potential_crop in crop_list:
        if f'_{potential_crop}' in shapefile_path:
            crop = potential_crop
            break
    return crop

def main():
    parser = argparse.ArgumentParser(description='Process to extract plot tiles.')
    parser.add_argument('--csv_folder_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--image_folder_path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--shapefile_path1', type=str, required=True, help='Path to the soy shapefile')
    parser.add_argument('--shapefile_path2', type=str, required=True, help='Path to the corn shapefile')
    parser.add_argument('--output_path1', type=str, required=True, help='Path to the output folder for soy')
    parser.add_argument('--output_path2', type=str, required=True, help='Path to the output folder for corn')
    parser.add_argument('--crop1', type=str, required=True, help='Crop1')
    parser.add_argument('--crop2', type=str, required=True, help='Crop2')
    parser.add_argument('--location', type=str, required=True, help='Location name')
    parser.add_argument('--date', type=str, required=True, help='Date of the images')
    parser.add_argument('--plotimage_source', type=str, help='OM or IR')
    
    args = parser.parse_args()
    
    csv_folder_path = args.csv_folder_path
    image_folder_path = args.image_folder_path
    shapefile_path1 = args.shapefile_path1
    shapefile_path2 = args.shapefile_path2
    output_path1 = args.output_path1
    output_path2 = args.output_path2
    location = args.location
    date = args.date
    plotimage_source = args.plotimage_source

    CSV_DIR = Path(args.csv_folder_path)
    print(f'csv dir: {CSV_DIR}')
    # Find the *geotags.csv files in the directory
    csv_files = list(CSV_DIR.glob('*geotags.csv'))

    if csv_files:
        try:
            df_csv = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
            if df_csv.empty:
                print(f'Empty CSV file found: {csv_files[0]}')
                csv_files = []
        except pd.errors.EmptyDataError:
            print(f'CSV file exists but has no data: {csv_files[0]}')
            csv_files = []
        except Exception as e:
            print(f'Error reading CSV file {csv_files[0]}: {str(e)}')
            csv_files = []


    if not csv_files:
        print("No geotags.csv file found")


    os.makedirs(output_path1, exist_ok=True)
    os.makedirs(output_path2, exist_ok=True)

    if plotimage_source == "om":
        print(f'Processing OM images')
        #os.makedirs(os.path.join(output_path_soy, "plot_tiles_ms_om"), exist_ok=True)
        #os.makedirs(os.path.join(output_path_soy, "plot_tiles_rgb_om"), exist_ok=True)

        if shapefile_path1:
            if os.path.exists(shapefile_path1):
                print(f'Processing shapefile1: {shapefile_path1}')
                crop = args.crop1
                print(f'Crop: {crop}')
                if crop:
                    output_filenames_om1 = extract_plot_tiles_om_parallel(shapefile_path1, output_path1, image_folder_path, crop, location, date)
                    # Remove directory if empty
                    if not os.listdir(output_path1):
                        os.rmdir(output_path1)
                else:
                    print(f'No crop found in shapefile path 1: {shapefile_path1}')
            else:
                print(f'Shapefile path 1 does not exist: {shapefile_path1}')
        if shapefile_path2:
            if os.path.exists(shapefile_path2):
                print(f'Processing shapefile2: {shapefile_path2}')
                crop = args.crop2
                print(f'Crop: {crop}')
                if crop:
                    output_filenames_om2 = extract_plot_tiles_om_parallel(shapefile_path2, output_path2, image_folder_path, crop, location, date)
                    # Remove directory if empty
                    if not os.listdir(output_path2):
                        os.rmdir(output_path2)
                else:
                    print(f'No crop found in shapefile path 2: {shapefile_path2}')
            else:
                print(f'Shapefile path 2 does not exist: {shapefile_path2}')

    elif plotimage_source in ["dgr", "ir"]:
        sorted_raster_files = get_sorted_raster_files(df_csv, image_folder_path)
       
        if shapefile_path1:
            if os.path.exists(shapefile_path1):
                print(f'Processing shapefile1: {shapefile_path1}')
                crop = args.crop1
                print(f'Crop: {crop}')
                if crop:
                    output_filenames_ir1 = extract_plot_tiles_ir_parallel(shapefile_path1, output_path1, sorted_raster_files, crop, location, date)
                    # Remove directory if empty
                    if not os.listdir(output_path1):
                        os.rmdir(output_path1)
                else:
                    print(f'No crop found in shapefile path 1: {shapefile_path1}')
            else:
                print(f'Shapefile path 1 does not exist: {shapefile_path1}')
        if shapefile_path2:
            if os.path.exists(shapefile_path2):
                print(f'Processing shapefile2: {shapefile_path2}')
                crop = args.crop2
                print(f'Crop: {crop}')
                if crop:
                    output_filenames_ir2 = extract_plot_tiles_ir_parallel(shapefile_path2, output_path2, sorted_raster_files, crop, location, date)
                    # Remove directory if empty
                    if not os.listdir(output_path2):
                        os.rmdir(output_path2)
                else:
                    print(f'No crop found in shapefile path 2: {shapefile_path2}')
            else:
                print(f'Shapefile path 2 does not exist: {shapefile_path2}')

if __name__ == "__main__":
    main()
