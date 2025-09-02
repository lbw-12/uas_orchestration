import argparse
import rasterio
from rasterio.transform import from_origin
import pandas as pd
import os
from pyproj import Transformer, CRS
import numpy as np
from affine import Affine
import math
import cv2
from rasterio.transform import from_bounds
import time

import sys
import shutil
import pyproj
from pathlib import Path
import glob



def get_elevation(lat, long, elevation_map_path):
    print(lat, long, elevation_map_path, "stuff")
    possible_elevations = os.listdir(elevation_map_path)
    for elevation_map in possible_elevations:
        print(elevation_map)
        if ".tif" not in elevation_map and ".img" not in elevation_map:
            continue
        with rasterio.open(elevation_map_path + elevation_map, sharing=False) as dataset:
            print("made it here somehow")
            transformer = pyproj.Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
            x, y = transformer.transform(long, lat)
            print(x, y, "x, y")
            row, col = dataset.index(x, y)

            if min(row, col) < 0 or max(row, col) > dataset.width-1:  # works because tiles are square
                print(row, col)
                continue

            elevation = dataset.read(1)[row, col]

            return elevation / 3.281
    print("unable to find elevation at ", lat, long)

"""def get_elevation(latitude, longitude):
    url = f'https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}'

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            if 'results' in data and len(data['results']) > 0:
                elevation = data['results'][0]['elevation']

                return elevation
            else:
                print('No elevation data found.')
        else:
            print('Request failed.')
            return None

    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')"""

def get_geoid(location):
    # for new locations, find  the difference betweeen geoid and ellipsoid height here, use egm2008-5:
    # https://www.unavco.org/software/geodetic-utilities/geoid-height-calculator/geoid-height-calculator.html


    if location == 'frantom':
        geoid_height = -33.71
    elif location == 'northwest':
        geoid_height = -36.13
    elif location == 'western':
        geoid_height = -33.67
    elif location == 'wooster':
        geoid_height = -33.71
    elif location == 'f7':
        geoid_height = -33.68
    else:
        print(location)
        location = 'unknown'
        print('Location not found.')
        exit()

    return geoid_height

def get_camera_intrinsics(sensor):
    if sensor == 'sonyrgb61':
        sensor_width_mm = 35.7
        sensor_height_mm = 23.3
        focal_length = 24.0

    if sensor == 'sonyrx1rii':
        sensor_width_mm = 35.0
        sensor_height_mm = 23.3
        focal_length = 32.8

    if sensor == 'altum':
        sensor_width_mm = 7.12
        sensor_height_mm = 5.33
        focal_length = 8

    return sensor_width_mm, sensor_height_mm, focal_length

def get_geotags(df, file_name):
        file_row = df[df[df.columns[0]] == file_name]
        if file_row.empty:
            print(f'No geotags found for {file_name}')
            return None
        file_row = file_row.iloc[0]
        center_lat = file_row[df.columns[1]]
        center_lon = file_row[df.columns[2]]
        altitude = file_row[df.columns[3]]
        omega = file_row[df.columns[4]]
        phi = file_row[df.columns[5]]
        kappa = file_row[df.columns[6]]
        return center_lat, center_lon, altitude, omega, phi, kappa

def get_sensor(base_dir):

    folder = base_dir.split('/')[-2]

    if 'sony' in folder.lower() and '2024' in folder.lower():
        sensor = 'sonyrgb61'
    elif 'sony' in folder.lower() and '2023' in folder.lower():
        sensor = 'sonyrx1rii'
    elif 'altum' in folder.lower():
        sensor = 'altum'
    else:
        sensor = 'unknown'

    return sensor

def get_crs_zone(df):
    file_row = df.iloc[0]
    center_lon = file_row[df.columns[2]]
    crs_zone = int(math.floor((center_lon + 180) / 6) + 1)
    crs = f'EPSG:326{crs_zone}'
    return crs

def get_location(folder):
    print(folder)
    if 'frantom' in folder.lower():
        location = 'frantom'
    elif 'northwest' in folder.lower():
        location = 'northwest'
    elif 'western' in folder.lower():
        location = 'western'
    elif 'wooster' in folder.lower():
        location = 'wooster'
    elif 'f7' in folder.lower():
        location = 'f7'
    elif "nrs" in folder.lower():
        location = "northwest"
    else:
        location = 'unknown'
    return location

def get_adjusted_coordinates(center_lat, center_lon, altitude, omega_deg, phi_deg, kappa_deg, img_width, img_height,
                             sensor, crs, mode='silent'):
    omega = -np.radians(omega_deg)
    phi = -np.radians(phi_deg)
    kappa = -np.radians(kappa_deg)
    R_x = np.array([[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]])
    R_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    R_z = np.array([[np.cos(kappa), np.sin(kappa), 0], [-np.sin(kappa), np.cos(kappa), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x

    sensor_width_mm, sensor_height_mm, focal_length = get_camera_intrinsics(sensor)

    sensor_diagonal = math.sqrt(sensor_width_mm ** 2 + sensor_height_mm ** 2)
    fov = 2 * np.arctan(sensor_diagonal / (2 * focal_length))
    sx = focal_length * (img_width / 2) / np.tan(math.radians(fov / 2))
    sy = focal_length * (img_height / 2) / np.tan(math.radians(fov / 2))
    cx = img_width / 2
    cy = img_height / 2
    K = np.array([[sx, 0, cx], [0, sy, cy], [0, 0, 1]])

    Z_transformed = R @ np.array([0, 0, 1])
    elev_hyp = altitude / Z_transformed[2]

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    t = np.array([[center_x], [center_y], [altitude]])
    R_t = np.concatenate((R, t), axis=1)
    M = np.dot(K, R_t)
    real_loc = np.array([[center_x], [center_y], [-altitude]])
    transformed_loc = np.dot(np.linalg.pinv(M), real_loc)

    ground_x = center_x + transformed_loc[0][0]
    ground_y = center_y + transformed_loc[1][0]

    if mode == 'verbose':
        print(f'omega deg: {omega_deg}')
        print(f'omega: {omega}')
        print(f'phi deg: {phi_deg}')
        print(f'phi: {phi}')
        print(f'kappa deg: {kappa_deg}')
        print(f'kappa: {kappa}')
        print(f'R_x: \n{R_x}')
        print(f'R_y: \n{R_y}')
        print(f'R_z: \n{R_z}')
        print(f'R calculated from phi, omega, kappa: \n{R}')
        print(f'ground_x: {ground_x}')
        print(f'ground_y: {ground_y}')
        print(f'elev_hyp: {elev_hyp}')
        print(f'center_x: {center_x}')
        print(f'center_y: {center_y}')

    # return adjusted_lon, adjusted_lat, R
    return ground_x, ground_y, R, elev_hyp


def georeferencing(file_name, df, location, crs, input_dir, output_path, sensor, mode, elevation_map_path, geoid_height):
    input_filename = os.path.join(input_dir, file_name)
    output_filename = os.path.join(output_path, f'dgr_{os.path.splitext(file_name)[0]}.tif')

    if os.path.exists(input_filename):
        if mode == 'verbose':
            print(f"reading input_filename: {input_filename}")
        try:
            with rasterio.open(input_filename) as dataset:

                img_width, img_height = dataset.width, dataset.height

                list_of_attributes = get_geotags(df, file_name)
                if list_of_attributes is None:
                    return

                center_lat, center_lon, altitude_ell, omega, phi, kappa = list_of_attributes

                ground_elevation = get_elevation(center_lat, center_lon, elevation_map_path)
                if ground_elevation is None:
                    return
                #geoid_height = get_geoid(location)
                
                adj_factor = 1.8  # Magic number to adjust elevation in meters to improve spatial scale
                print(f'altitude_ell: {altitude_ell}')
                print(f'geoid height: {geoid_height}')
                print(f'ground elevation: {ground_elevation}')
                print(f'adj_factor: {adj_factor}')
                altitude_agl_m = altitude_ell - geoid_height - ground_elevation + adj_factor

                print(f'altitude_agl_m: {altitude_agl_m}')

                sensor_width_mm, sensor_height_mm, focal_length = get_camera_intrinsics(sensor)

                pixel_size_x_meters = sensor_width_mm / img_width
                pixel_size_y_meters = sensor_height_mm / img_height

                # Ground sampling distance (GSD) in meters/pixel
                gsd_x = pixel_size_x_meters * (altitude_agl_m / focal_length)
                gsd_y = pixel_size_y_meters * (altitude_agl_m / focal_length)

                center_x, center_y, R, elev_hyp = get_adjusted_coordinates(center_lat, center_lon, -altitude_agl_m, omega,
                                                                           phi, kappa, img_width, img_height, sensor, crs,
                                                                           mode=mode)

                half_width_m = (img_width * gsd_x) / 2
                half_height_m = (img_height * gsd_y) / 2

                image_orig = dataset.read()

                # Create a new band with all values set to 255 and append to existing data
                binary_mask = np.full((1, image_orig.shape[1], image_orig.shape[2]), 255, dtype=image_orig.dtype)
                image_bmask = np.vstack((image_orig, binary_mask))

                if image_bmask.dtype != np.uint8 and sensor == 'sony':
                    image_bmask = image_bmask.astype(np.uint8)

                corners_3d = np.array([
                    [-half_width_m, half_height_m, -elev_hyp],  # Top-left
                    [half_width_m, half_height_m, -elev_hyp],  # Top-right
                    [half_width_m, -half_height_m, -elev_hyp],  # Bottom-right
                    [-half_width_m, -half_height_m, -elev_hyp]  # Bottom-left
                ])

                # Apply rotation to each corner
                rotated_corners = np.dot(R,
                                         corners_3d.T).T  # Transpose for matrix multiplication, order bottom left, top left, top right, bottom right

                # Project each corner onto the ground plane
                projected_corners = np.array([
                    [center_x + (corner[0] / -corner[2]) * elev_hyp,
                     center_y + (corner[1] / -corner[2]) * elev_hyp]
                    for corner in rotated_corners
                ])

                # Resulting corner coordinates after applying rotation and projecting to 2D
                new_top_left = projected_corners[0]
                new_top_right = projected_corners[1]
                new_bottom_right = projected_corners[2]
                new_bottom_left = projected_corners[3]

                # Define original corner points of the untransformed image
                original_corners = np.array([
                    [(center_x - half_width_m), (center_y + half_height_m)],  # Top-left
                    [(center_x + half_width_m), (center_y + half_height_m)],  # Top-right
                    [(center_x + half_width_m), (center_y - half_height_m)],  # Bottom-right
                    [(center_x - half_width_m), (center_y - half_height_m)]  # Bottom-left
                ])

                # Define the destination corner points after accounting for pitch and roll
                # These would be adjusted based on your calculations for phi and omega effects
                destination_corners = np.array([
                    [new_top_left[0], new_top_left[1]],
                    [new_top_right[0], new_top_right[1]],
                    [new_bottom_right[0], new_bottom_right[1]],
                    [new_bottom_left[0], new_bottom_left[1]]
                ])

                # Create the affine transformation from UTM to pixels
                transform = Affine.translation(original_corners[0][0], original_corners[0][1]) * Affine.scale(gsd_x, -gsd_y)
                inv_transform = ~transform

                # Convert each corner from UTM to pixel coordinates using the inverse transform
                original_corners_pixels = np.array([inv_transform * (x, y) for x, y in original_corners], dtype=np.float32)
                destination_corners_pixels = np.array([inv_transform * (x, y) for x, y in destination_corners],
                                                      dtype=np.float32)

                # Calculate the homography matrix
                homography_matrix, _ = cv2.findHomography(original_corners_pixels, destination_corners_pixels)

                # if sensor == "sony":
                image_bmask = image_bmask.transpose(1, 2, 0)  # Convert to (height, width, channels)

                # the following code used to convert to bgr and rgb, even though opencv expects a bgra image,
                # it doesn't actually matter for affine transformations

                """try:
                    image_bgr = cv2.cvtColor(image_bmask, cv2.COLOR_RGBA2BGRA)
                except cv2.error:
                    print(f'Error converting image: {input_filename}')
                    return"""
                # Calculate the bounding box for the destination corners in pixel coordinates
                min_x_px = np.min(destination_corners_pixels[:, 0])
                max_x_px = np.max(destination_corners_pixels[:, 0])
                min_y_px = np.min(destination_corners_pixels[:, 1])
                max_y_px = np.max(destination_corners_pixels[:, 1])

                # Calculate the new dimensions for the output image
                output_width_px = int(np.ceil(max_x_px - min_x_px))
                output_height_px = int(np.ceil(max_y_px - min_y_px))

                # Offset the homography to adjust for new bounds
                # This adjustment aligns the top-left of the output image to (min_x, min_y)
                translation_matrix = np.array([[1, 0, -min_x_px],
                                               [0, 1, -min_y_px],
                                               [0, 0, 1]])

                # Adjust the homography matrix by multiplying with the translation
                adjusted_homography = translation_matrix @ homography_matrix

                # Apply the adjusted homography to warp the image onto the resized canvas
                transformed_image = cv2.warpPerspective(image_bmask, adjusted_homography, (output_width_px, output_height_px))
                """try:
                    # Convert BGR to RGB for correct color display
                    transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGRA2RGBA)
                except cv2.error:
                    print(f'Error converting image: {input_filename}')
                    return"""


                # Calculate min_x, min_y, max_x, and max_y dynamically based on all four corners in UTM coordinates
                min_x_m = min(destination_corners[:, 0])
                max_x_m = max(destination_corners[:, 0])
                min_y_m = min(destination_corners[:, 1])
                max_y_m = max(destination_corners[:, 1])

                # Create an affine transformation based on the new bounds and image size
                transform = from_bounds(min_x_m, min_y_m, max_x_m, max_y_m, output_width_px, output_height_px)

                transformed_image_rasterio = np.transpose(transformed_image,
                                                          (2, 0, 1))  # Convert to (bands, height, width)

                if sensor == "altum":
                    count = 2
                else: # code in use here is for channels
                    count = 4
                print(count, sensor)

                # Write the transformed image to a GeoTIFF
                print(f'output_filename: {output_filename}')
                with rasterio.open(
                        output_filename,
                        "w",
                        driver="GTiff",
                        height=output_height_px,
                        width=output_width_px,
                        count=count,
                        dtype=transformed_image.dtype,
                        crs=crs,
                        transform=transform,
                        compress="deflate"  # Optional compression
                ) as dst:
                    dst.write(transformed_image_rasterio)

                if mode == 'verbose':
                    print(
                        f"input_filename: {input_filename}, center_lat: {center_lat}, center_lon: {center_lon}, altitude_ell: {altitude_ell}, omega: {omega}, phi: {phi}, kappa: {kappa}")
                    print(f"altitude_agl_meters: {altitude_agl_m}")
                    print(f'ground elevation: {ground_elevation}')
                    print(f"gsd_x: {gsd_x}, gsd_y: {gsd_y}")
                    print(f"half_width_m: {half_width_m}, half_height_m: {half_height_m}")
                    print(f"center_x: {center_x}, center_y: {center_y}")
                    print(f"image_orig.shape: {image_orig.shape}")
                    print(f"image_bmask.shape: {image_bmask.shape}")
                    print(f'elev_hyp: {elev_hyp}')
                    print(f'corners_3d: \n{corners_3d}')
                    print(f'rotated_corners: \n{rotated_corners}')
                    print(f'projected_corners: \n{projected_corners}')
                    print(f'original_corners: \n{original_corners}')
                    print(f'destination_corners: \n{destination_corners}')
                    print(f'Shape of original image: {image_orig.shape}')
                    print(f' homography_matrix: {homography_matrix}')
                    print(f' homography_matrix.shape: {homography_matrix.shape}')
                    print(f'img_width: {img_width}, img_height: {img_height}')
                    print(f"Original corners in pixels: {original_corners_pixels}")
                    print(f"Destination corners in pixels: {destination_corners_pixels}")
                    print(f'original corners in UTM top left x: {original_corners[0][0]}')
                    print(f'original corners in UTM top left y: {original_corners[0][1]}')
                    print(f'shape of transformed_image_rgb: {transformed_image.shape}')
                    print(f'min_x: {min_x_m}, min_y: {min_y_m}, max_x: {max_x_m}, max_y: {max_y_m}')
                    print(f'transform = \n{transform}')
        except rasterio.errors.RasterioIOError as e:
            print("Error reading file: ", input_filename)
            print(e)
    else:
        print(f'File not found: {input_filename}')


def main():
    parser = argparse.ArgumentParser(description='Process images in a folder.')
    parser.add_argument('--input_folder', required=True, help='Input folder containing images')
    parser.add_argument('--output_folder', required=True, help='Output folder for processed images')
    args = parser.parse_args()

    # Get the necessary information for processing
    args_to_pass = collect_info_specific_path(args.input_folder, args.output_folder)
    if not args_to_pass:
        print("Failed to collect necessary information")
        return

    df_csv, input_dir, output_dir, sensor, elevation_map_path = args_to_pass
    df = df_csv
    crs = get_crs_zone(df)
    location = get_location(input_dir)

    # Get list of all image files
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.tif')]
    print(f"Found {len(jpg_files)} images to process")

    # Clean output directory
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if not file.startswith('._') and not file.startswith("output.txt") and (file.endswith('.tif') or file.lower().endswith('.jpg')):
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    else:
        os.makedirs(output_dir)

    start_time = time.time()

    # Process images sequentially
    for file_name in jpg_files:
        try:
            georeferencing(
                file_name,
                df,
                location,
                crs,
                input_dir,
                output_dir,
                sensor,
                "silent",
                elevation_map_path
            )
            print(f"Successfully processed {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    end_time = time.time()
    print(f"Total time consumed: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(jpg_files)} images")


def collect_info_specific_path(filepath, output_folder, elevation_folder):
    # just going to presume some things
    """location_of_elevation_tiles = "/fs/ess/PAS2699/nitrogen/data/uas/elevation_tiles/"
    western_elevation_map_path = location_of_elevation_tiles + "western/"
    wooster_elevation_map_path = location_of_elevation_tiles + "wooster/"
    frantom_elevation_map_path = location_of_elevation_tiles + "frantom/"
    northwest_elevation_map_path = location_of_elevation_tiles + "northwest/"
    f7_elevation_map_path = location_of_elevation_tiles + "f7/"
    kuntz_elevation_map_path = location_of_elevation_tiles + "kuntz/"
    print(f'filepath: {filepath}')
    flight = filepath.split("/")[-3]
    print(f'flight: {flight}')
    elevation_map_path = ""
    
    if "western" in flight.lower():
        elevation_map_path = western_elevation_map_path
        # shp_files = shp_files_western
    elif "wooster" in flight.lower():
        elevation_map_path = wooster_elevation_map_path
        # shp_files = shp_files_wooster
    elif "frantom" in flight.lower():
        elevation_map_path = frantom_elevation_map_path
        # shp_files = shp_files_frantom
    elif "northwest" in flight.lower():
        elevation_map_path = northwest_elevation_map_path
        # shp_files = shp_files_northwest
    elif "farmscience" in flight.lower() or "fsr" in flight.lower():
        elevation_map_path = f7_elevation_map_path
        # shp_files = shp_files_f7
    elif "kuntz" in flight.lower():
        elevation_map_path = kuntz_elevation_map_path
        # shp_files = shp_files_kuntz"""
    """else:
        print(flight.lower(), " did not contain any known pattern")"""
    
    elevation_map_path = elevation_folder
    
        # raise ValueError("Flight pattern not found")
    if not os.path.isdir(filepath):
        print(f'filepath: {filepath} is not a directory')
        return
    if "altum" in filepath.lower():
        MODE = "altum"
    elif "sony" in filepath.lower():
        if "2023" in filepath.lower():
            MODE = "sonyrx1rii"
        else:
            MODE = "sonyrgb61"
    else:
        print(f'sensor: {filepath.lower()} not found')
        return

    """if not os.path.isfile(filepath + "/01_Images/" + flight + "/OUTPUT/" + flight + " report.pdf"):
        print(f'report.pdf not found for {filepath}')
        return"""

    #INPUT_DIR = filepath + "/01_Images/" + flight + "/OUTPUT/"
    INPUT_DIR = filepath
    if not os.path.isdir(INPUT_DIR):
        print(f'INPUT_DIR: {INPUT_DIR} is not a directory')
        return

    #OUTPUT_PATH_GEO = filepath + "/01_Images/" + flight + "/DGR/"
    OUTPUT_PATH_GEO = output_folder

    # Go up a level to get the geotags.csv file
    #CSV_DIR = Path(INPUT_DIR).parent / 'OUTPUT'
    CSV_DIR = Path(INPUT_DIR)
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
        return
    CSV_FILE = csv_files[0]
    print(f'csv file: {CSV_FILE}')

    print(f'csv_files: {csv_files}, \nINPUT_DIR: {INPUT_DIR}, \nOUTPUT_PATH_GEO: {OUTPUT_PATH_GEO}, \nMODE: {MODE}, \nelevation_map_path: {elevation_map_path}')

    return df_csv, INPUT_DIR, OUTPUT_PATH_GEO, MODE, elevation_map_path




if __name__ == "__main__":
    main()

