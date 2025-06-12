import geopandas as gpd
from shapely.geometry import Point, shape, mapping, box
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

def get_plot_tile(row, src, plot_no, date, location, sensor):
    buffered_geometry = row["geometry"].buffer(5 * abs(src.transform.a)).simplify(0.5)
    shapes = [mapping(buffered_geometry)]
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, filled=True, all_touched=True, nodata=None)
    shape_coords = shapes[0]['coordinates']
    angle = get_longest_edge_angle(shape_coords, out_transform)
    bands, height, width = out_image.shape
    center = (width / 2, height / 2)
    rotM = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    rotated_bands = []
    for i in range(bands):
        image_band = out_image[i]  
        rotated_band = cv2.warpAffine(src=image_band, M=rotM, dsize=(width, height))
        rotated_bands.append(rotated_band)
    
    rotated_image = np.array(rotated_bands, dtype=np.float32)
    cropped_image, has_valid_band = crop_black_white_pixels(rotated_bands, plot_no, date, location, sensor)
    final_cropped = np.array(cropped_image, dtype=np.float32)

    return final_cropped, has_valid_band, out_transform

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
        if shpfile.crs.to_epsg() != src.crs.to_epsg():
            raise ValueError("Shapefile and OM CRS do not match.", shapefile_path, image_file)
        raster_center = [(src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2]
        print(f'Length of shapefile: {len(shpfile)}')
        
        for index, row in shpfile.iterrows():
            if "plot" not in row:
                print(f'Plot not found in shapefile: {row}')
                continue
            plot_no = row["plot"]
            plot_center = row["geometry"].centroid.coords[0]
            distance_to_image_center = calculate_distance(plot_center, raster_center)
            if plot_no in processed_plots_info:
                continue
            if intersects(row, src.bounds):
                final_cropped, has_valid_band, out_transform = get_plot_tile(row, src, plot_no, date, location, sensor)
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
            for index, row in shpfile.iterrows():
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
                    final_cropped, has_valid_band, out_transform = get_plot_tile(row, src, plot_no, date, location, sensor)
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
                final_cropped, has_valid_band, out_transform = get_plot_tile(row, src, plot_no, date, location, sensor)
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
def get_sorted_raster_files(csv_file_path, image_folder_path):
    omega_phi_list = []
    with open(csv_file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            filename_in_csv = row[0]
            omega_value = float(row[4])
            phi_value = float(row[5])
            kappa_value = float(row[6])

            match = re.search(r"(\d+_\d+)", filename_in_csv)
            num_part_csv = match.group(1)
            
            for image_filename in os.listdir(image_folder_path):
                if image_filename.lower().endswith(".tif"):
                    match = re.search(r"(\d+_\d+)", image_filename)
                    num_part_image = match.group(1)
                    if num_part_csv == num_part_image:
                        output_filename = os.path.join(image_folder_path, image_filename)
                        omega_phi_list.append((output_filename, omega_value, phi_value, kappa_value))
                        break
    sorted_raster_files = [item[0] for item in sorted(omega_phi_list, key=lambda x: abs(x[1]) + abs(x[2]))]

    return sorted_raster_files
def main():
    parser = argparse.ArgumentParser(description='Process to extract plot tiles.')
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--image_folder_path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--shapefile_path_soy', type=str, required=True, help='Path to the soy shapefile')
    parser.add_argument('--shapefile_path_corn', type=str, required=True, help='Path to the corn shapefile')
    parser.add_argument('--output_path_soy', type=str, required=True, help='Path to the output folder for soy')
    parser.add_argument('--output_path_corn', type=str, required=True, help='Path to the output folder for corn')
    parser.add_argument('--location', type=str, required=True, help='Location name')
    parser.add_argument('--date', type=str, required=True, help='Date of the images')
    parser.add_argument('--plotimage_source', type=str, help='OM or IR')
    
    args = parser.parse_args()
    
    csv_file_path = args.csv_file_path
    image_folder_path = args.image_folder_path
    shapefile_path_soy = args.shapefile_path_soy
    shapefile_path_corn = args.shapefile_path_corn
    output_path_soy = args.output_path_soy
    output_path_corn = args.output_path_corn
    location = args.location
    date = args.date
    plotimage_source = args.plotimage_source

    if plotimage_source == "om":
        print(f'Processing OM images')
        #os.makedirs(os.path.join(output_path_soy, "plot_tiles_ms_om"), exist_ok=True)
        #os.makedirs(os.path.join(output_path_soy, "plot_tiles_rgb_om"), exist_ok=True)
        os.makedirs(output_path_soy, exist_ok=True)
        os.makedirs(output_path_corn, exist_ok=True)
        if shapefile_path_soy:
            print(f'Processing soy shapefile: {shapefile_path_soy}')
            output_filenames_om_soy = extract_plot_tiles_om(shapefile_path_soy, output_path_soy, image_folder_path, "soy", location, date)
        if shapefile_path_corn:
            print(f'Processing corn shapefile: {shapefile_path_corn}')
            output_filenames_om_corn = extract_plot_tiles_om(shapefile_path_corn, output_path_corn, image_folder_path, "corn", location, date)

    elif plotimage_source == "dgr":
        sorted_raster_files = get_sorted_raster_files(csv_file_path, image_folder_path)
       
        os.makedirs(os.path.join(output_path_soy, "plot_tiles_rgb_dgr"), exist_ok=True)
        os.makedirs(os.path.join(output_path_soy, "plot_tiles_ms_dgr"), exist_ok=True)
        
        output_filenames_soy = extract_plot_tiles_ir(shapefile_path_soy, output_path_soy, sorted_raster_files, 'soy', location, date)
        output_filenames_corn = extract_plot_tiles_ir(shapefile_path_corn, output_path_corn, sorted_raster_files, 'corn', location, date)
    elif plotimage_source == "ir":
        print("Processing IR images")
        sorted_raster_files = get_sorted_raster_files(csv_file_path, image_folder_path)
       
        os.makedirs(os.path.join(output_path_soy, "plot_tiles_rgb_ir"), exist_ok=True)
        os.makedirs(os.path.join(output_path_soy, "plot_tiles_ms_ir"), exist_ok=True)
        
        output_filenames_soy = extract_plot_tiles_ir(shapefile_path_soy, output_path_soy, sorted_raster_files, 'soy', location, date)
        output_filenames_corn = extract_plot_tiles_ir(shapefile_path_corn, output_path_corn, sorted_raster_files, 'corn', location, date)
if __name__ == "__main__":
    main()
