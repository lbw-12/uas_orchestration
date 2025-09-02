import os
import shutil
import numpy as np
from os import listdir
import cv2 as cv
import time
import pickle
import copy
import rasterio
import pyproj
from rasterio.warp import reproject, Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
import shapefile
import argparse
from sklearn.cluster import DBSCAN
import time
from pathlib import Path
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor



"""plt.rcParams['figure.figsize'] = [16, 10]
np.set_printoptions(linewidth=100)
file_dir = 'C:\\Users\\Ryan Waltz\\Desktop\\Python\\agcv\\drawinglines\\timelapse\\timelapse\\20230523-wooster418cornall\\20230523-testimages\\'

folder_elements = sorted(listdir(file_dir))

full_path = [file_dir + _ for _ in folder_elements]


image_filenames = []
imageonly_filenames = []"""

"""for i in range(len(full_path)):
    if (full_path[i].split('.')[-1] in ['JPG', "jpg"]):
        image_filenames.append(full_path[i])
        imageonly_filenames.append(folder_elements[i])"""


class ConfidenceError(Exception):
    def __init__(self, value):
        self.value = value


def return_bad_pts(loc1, loc2, loc3, loc4):
    x_coords = [loc1[0], loc2[0], loc3[0], loc4[0]]
    y_coords = [loc1[1], loc2[1], loc3[1], loc4[1]]
    mean_x = sum(x_coords) / len(x_coords)
    mean_y = sum(y_coords) / len(y_coords)
    list_of_problematic_entries = []
    for i in x_coords:
        if abs(i - mean_x) > 10:
            list_of_problematic_entries.append(x_coords.index(i))

    for i in y_coords:
        if abs(i - mean_y) > 10:
            if i not in list_of_problematic_entries:
                list_of_problematic_entries.append(y_coords.index(i))
    return list_of_problematic_entries


def match(patch, template, locn, search, index, debug=False, file2=None):
    # Calculate mean values of red, green, and blue for the template image
    mean_t = np.zeros(3)
    mean_t[0] = np.average(template[:, :, 0])
    mean_t[1] = np.average(template[:, :, 1])
    mean_t[2] = np.average(template[:, :, 2])

    # Calculate standard variation for the template image
    sd_t = np.zeros(3)
    sd_t[0] = (np.var(template[:, :, 0], ddof=1)) ** .5
    sd_t[1] = (np.var(template[:, :, 1], ddof=1)) ** .5
    sd_t[2] = (np.var(template[:, :, 2], ddof=1)) ** .5

    # Initialize mean, standard deviation, and ncc values for the patch image
    mean_p = np.zeros(3)
    sd_p = np.zeros(3)
    ncc = np.zeros(3)

    # Initialize an array to store all of the ncc values
    ncc_list = np.zeros((len(patch), 2))

    # Calculate ncc values
    for i in range(len(patch)):
        for j in range(3):
            mean_p[j] = np.average(patch[i][:, :, j])
            sd_p[j] = (np.var(patch[i][:, :, j], ddof=1)) ** .5
            ncc[j] = np.sum((patch[i][:, :, j] - mean_p[j]) * (template[:, :, j] - mean_t[j]) / (sd_t[j] * sd_p[j]))
            ncc[j] /= (len(search) * len(search[0]) - 1)
        ncc_list[i, :] = [i, np.sum(ncc)]

        ind = ncc_list[:, 1].argsort()[::-1][:500]
        ncc_sorted = ncc_list[ind]
        # locn_sorted = locn[ind]

    if debug:
        debug_dir = Path(file2).parent / "debug"
        os.makedirs(debug_dir, exist_ok=True)
        best_match = patch[int(ncc_sorted[0][0])]
        best_match_center = (best_match.shape[0] // 2, best_match.shape[1] // 2)
        best_match_annotated = best_match.copy()
        best_match_annotated[best_match_center[0], best_match_center[1]] = [0, 0, 255]

        # Annotate the template with a green dot of the highest rated patch location
        #template_center = (locn[int(ncc_sorted[0][0])][0] + best_match.shape[0] // 2, locn[int(ncc_sorted[0][0])][1] + best_match.shape[1] // 2)
        template_annotated = template.copy()
        #template_annotated[template_center[0], template_center[1]] = [0, 0, 255]
        template_center = (template.shape[0] // 2, template.shape[1] // 2)
        template_annotated[template_center[0], template_center[1]] = [0, 0, 255]

        print(f'best_match: {template_center}')
        print(f'location: {locn[int(ncc_sorted[0][0])]}')

        best_match_annotated_bgr = cv.cvtColor(best_match_annotated, cv.COLOR_RGB2BGR)  
        template_annotated_bgr = cv.cvtColor(template_annotated, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(debug_dir, f"best_match_patch_{index}.png"), best_match_annotated_bgr)
        cv.imwrite(os.path.join(debug_dir, f"best_match_template_{index}.png"), template_annotated_bgr)

    try:
        return patch[int(ncc_sorted[0][0])], locn[int(ncc_sorted[0][0])], ncc_sorted[0][0]
    except UnboundLocalError:
        return None, None, None


def return_all_pts_from_shapefile_old(shapefile_path, size, search_size_m, patch_size_m, file1, file2):
    pts = []
    if os.path.exists(shapefile_path):
        with shapefile.Reader(shapefile_path) as shp:

            fields = [field[0] for field in shp.fields[1:]]
            try:
                id_field_name = 'id'
                id_index = fields.index(id_field_name) 
            except ValueError:
                print(f"Error: Field '{id_field_name}' not found in shapefile!")
                print(f"Available fields are: {fields}")
                # Exit or handle the error appropriately
                exit()

            for shape_rec in shp.iterShapeRecords():
                point_id = shape_rec.record[id_index]
                point_coords = shape_rec.shape.points[0]
                pts.append(point_coords)
                print(f'point: {point_id}, {point_coords}')

    pts_template = []
    with rasterio.open(file1) as dataset:
        for pt in pts:
            new_pt = [pt[0] - patch_size_m / 2, pt[1] - patch_size_m / 2] # Gets the top left corner of the path
            pts_template.append(new_pt)
        tl_right_corner = [pts_template[0][0] + patch_size_m / 2, pts_template[0][1] + patch_size_m / 2]
        tl_search_corner = [pts_template[0][0] - search_size_m, pts_template[0][1] - search_size_m]
        pts_template_pxl = []
        for pt in pts_template:
            x = int(dataset.index(pt[0], pt[1])[0] * size / 100)
            y = int(dataset.index(pt[0], pt[1])[1] * size / 100)
            pts_template_pxl.append((x, y))
        tl_right_corner = dataset.index(tl_right_corner[0], tl_right_corner[1])
        tl_search_corner = dataset.index(tl_search_corner[0], tl_search_corner[1])
        # scale tl_right_corner and tl_search_corner by size/100
        tl_right_corner = (int(tl_right_corner[0] * size / 100), int(tl_right_corner[1] * size / 100))
        tl_search_corner = (int(tl_search_corner[0] * size / 100), int(tl_search_corner[1] * size / 100))

        patchsize = int(abs(tl_right_corner[0] - pts_template_pxl[0][0]))
        search_size = int(abs(tl_search_corner[0] - pts_template_pxl[0][0]))
    with rasterio.open(file2) as dataset:

        pts_search_pxl = []
        for pt in pts_template:
            x = int(dataset.index(pt[0], pt[1])[0] * size / 100)
            y = int(dataset.index(pt[0], pt[1])[1] * size / 100)
            pts_search_pxl.append((x, y))

    return pts_template_pxl, pts_search_pxl, search_size, patchsize


def _read_shapefile_points(shapefile_path: str) -> list[tuple[float, float]]:
    """
    Reads all point coordinates from a shapefile.

    Args:
        shapefile_path: Path to the input shapefile.

    Returns:
        A list of (x, y) coordinate tuples.
    
    Raises:
        FileNotFoundError: If the shapefile does not exist.
        ValueError: If the shapefile does not contain point geometries.
    """
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found at: {shapefile_path}")

    geo_coords = []
    with shapefile.Reader(shapefile_path) as shp:
        if shp.shapeType != shapefile.POINT:
            raise ValueError("Shapefile must contain POINT geometries.")
            
        for shape_rec in shp.iterShapes():
            # For Point features, shape.points[0] gives the coordinate pair
            geo_coords.append(tuple(shape_rec.points[0]))
    
    print(f"Read {len(geo_coords)} points from {os.path.basename(shapefile_path)}")
    return geo_coords


def _geo_to_scaled_pixel(dataset: rasterio.DatasetReader, geo_coords: list[tuple[float, float]], scale_percent: int) -> list[tuple[int, int]]:
    """
    Converts a list of geographic coordinates to scaled pixel coordinates.

    Args:
        dataset: An open rasterio dataset.
        geo_coords: A list of (x, y) geographic coordinates.
        scale_percent: The percentage to scale the pixel coordinates by (e.g., 50).

    Returns:
        A list of (row, col) scaled pixel coordinate tuples.
    """
    pixel_coords = []
    scale_factor = scale_percent / 100.0
    for x, y in geo_coords:
        # dataset.index() converts geographic coordinates to pixel row, col
        row, col = dataset.index(x, y)
        scaled_row = int(row * scale_factor)
        scaled_col = int(col * scale_factor)
        pixel_coords.append((scaled_row, scaled_col))
    return pixel_coords


def _meters_to_pixels(dataset: rasterio.DatasetReader, size_in_meters: float, scale_percent: int) -> int:
    """
    Converts a size in meters to a size in pixels for a given raster and scale.

    Args:
        dataset: An open rasterio dataset.
        size_in_meters: The size to convert (e.g., patch width in meters).
        scale_percent: The percentage the raster will be scaled by.

    Returns:
        The equivalent size in pixels.
    """
    # dataset.transform.a gives the width of a pixel in CRS units (usually meters)
    pixel_width_in_meters = abs(dataset.transform.a)
    size_in_full_res_pixels = size_in_meters / pixel_width_in_meters
    
    # Adjust for the final scaled size of the image
    scale_factor = scale_percent / 100.0
    scaled_size_in_pixels = size_in_full_res_pixels * scale_factor
    return int(scaled_size_in_pixels)




import rasterio
import shapefile
import os
import numpy as np

# Make sure this function is defined in your script before find_match_refactored
def get_patch_and_search_locations(shapefile_path, raster_file_for_coords, patch_sidelength_m, search_buffer_m):
    """
    Reads points and their IDs from a shapefile and calculates all necessary pixel
    coordinates and sizes at full resolution.
    """
    print("\n--- Calculating Sizes and Locations from Shapefile ---")
    
    # This list will store dictionaries, each holding an ID and geographic coords
    geo_pts_with_ids = []
    id_field_name = 'id'  #<-- Make sure this matches your field name

    with rasterio.open(raster_file_for_coords) as src:
        gsd = src.res[0]
        patch_size_px = int(round(patch_sidelength_m / 2 / gsd))
        search_buffer_px = int(round(search_buffer_m / gsd))

        print(f"Detected GSD: {gsd:.4f} m/pixel")
        print(f"Calculated Patch Side Length: {patch_size_px} px")
        print(f"Calculated Search Buffer: {search_buffer_px} px")

        with shapefile.Reader(shapefile_path) as shp:
            fields = [field[0] for field in shp.fields[1:]]
            try:
                id_index = fields.index(id_field_name)
            except ValueError:
                raise ValueError(f"Field '{id_field_name}' not found in shapefile! Available fields: {fields}")

            for shape_rec in shp.iterShapeRecords():
                point_id = shape_rec.record[id_index]
                point_coords = shape_rec.shape.points[0]
                geo_pts_with_ids.append({'id': point_id, 'coords': point_coords})
        
        print(f"Read {len(geo_pts_with_ids)} points from {os.path.basename(shapefile_path)}")

        pts_template_pxl = []
        for pt_data in geo_pts_with_ids:
            geo_coords = pt_data['coords']
            center_row, center_col = src.index(geo_coords[0], geo_coords[1])
            tl_row = center_row - (patch_size_px // 2)
            tl_col = center_col - (patch_size_px // 2)
            pts_template_pxl.append((tl_row, tl_col))

    pts_search_pxl = pts_template_pxl
    
    # --- THIS IS THE KEY CHANGE ---
    # We now return 5 items, including the list with the IDs.
    return geo_pts_with_ids, pts_template_pxl, pts_search_pxl, search_buffer_px, patch_size_px

def return_pts_from_shapefile(shapefile_path, size, search_size_m, patch_size_m, file1, file2, top_left_idx,
                              top_right_idx, bottom_left_idx, bottom_right_idx):
    pts = []
    if os.path.exists(shapefile_path):
        with shapefile.Reader(shapefile_path) as shp:
            for shape in shp.shapes():
                pts.append(shape.points[0])

    with rasterio.open(file1) as dataset:

        top_left = [pts[top_left_idx][0] - patch_size_m / 2, pts[top_left_idx][1] - patch_size_m / 2]
        top_left_right_corner = [pts[top_left_idx][0] + patch_size_m / 2, pts[top_left_idx][1] - patch_size_m / 2]
        top_left_search_corner = [top_left[0] - patch_size_m / 2 - search_size_m, top_left[1] - patch_size_m / 2]
        top_right = [pts[top_right_idx][0] - patch_size_m / 2, pts[top_right_idx][1] - patch_size_m / 2]
        bottom_left = [pts[bottom_left_idx][0] - patch_size_m / 2, pts[bottom_left_idx][1] - patch_size_m / 2]
        bottom_right = [pts[bottom_right_idx][0] - patch_size_m / 2, pts[bottom_right_idx][1] - patch_size_m / 2]

        """top_left[1], top_left[0] = dataset.index(top_left[0], top_left[1])
        top_left_right_corner[1], top_left_right_corner[0] = dataset.index(top_left_right_corner[0], top_left_right_corner[1])
        top_left_search_corner[1], top_left_search_corner[0] = dataset.index(top_left_search_corner[0], top_left_search_corner[1])
        top_right[1], top_right[0] = dataset.index(top_right[0], top_right[1])
        bottom_left[1], bottom_left[0] = dataset.index(bottom_left[0], bottom_left[1])
        bottom_right[1], bottom_right[0] = dataset.index(bottom_right[0], bottom_right[1])"""

        top_left = dataset.index(top_left[0], top_left[1])
        top_left_right_corner = dataset.index(top_left_right_corner[0], top_left_right_corner[1])
        print(top_left_search_corner, "top_left_search_corner in utm coords")
        top_left_search_corner = dataset.index(top_left_search_corner[0], top_left_search_corner[1])
        top_right = dataset.index(top_right[0], top_right[1])
        bottom_left = dataset.index(bottom_left[0], bottom_left[1])
        bottom_right = dataset.index(bottom_right[0], bottom_right[1])

        # multiply above values by size/100
        top_left = int(top_left[0] * size / 100), int(top_left[1] * size / 100)
        top_left_right_corner = int(top_left_right_corner[0] * size / 100), int(top_left_right_corner[1] * size / 100)
        top_left_search_corner = int(top_left_search_corner[0] * size / 100), int(
            top_left_search_corner[1] * size / 100)
        top_right = int(top_right[0] * size / 100), int(top_right[1] * size / 100)
        bottom_left = int(bottom_left[0] * size / 100), int(bottom_left[1] * size / 100)
        bottom_right = int(bottom_right[0] * size / 100), int(bottom_right[1] * size / 100)

        search_size = int(-top_left_search_corner[1] + top_left[1])
        patchsize = int(top_left_right_corner[1] - top_left[1])
        print("patchsize", patchsize)
        print("search_size", search_size)

    with rasterio.open(file2) as dataset:
        top_left_search = pts[0][0] - patch_size_m / 2, pts[0][1] - patch_size_m / 2
        top_right_search = pts[1][0] - patch_size_m / 2, pts[1][1] - patch_size_m / 2
        bottom_left_search = pts[2][0] - patch_size_m / 2, pts[2][1] - patch_size_m / 2
        bottom_right_search = pts[3][0] - patch_size_m / 2, pts[3][1] - patch_size_m / 2

        top_left_search = dataset.index(top_left_search[0], top_left_search[1])
        top_right_search = dataset.index(top_right_search[0], top_right_search[1])
        bottom_left_search = dataset.index(bottom_left_search[0], bottom_left_search[1])
        bottom_right_search = dataset.index(bottom_right_search[0], bottom_right_search[1])

        top_left_search = int(top_left_search[0] * size / 100), int(top_left_search[1] * size / 100)
        top_right_search = int(top_right_search[0] * size / 100), int(top_right_search[1] * size / 100)
        bottom_left_search = int(bottom_left_search[0] * size / 100), int(bottom_left_search[1] * size / 100)
        bottom_right_search = int(bottom_right_search[0] * size / 100), int(bottom_right_search[1] * size / 100)

    print(top_left, "top_left")
    print(top_left_search, "top_left_search")
    print(top_right, "top_right")
    print(top_right_search, "top_right_search")
    print(bottom_left, "bottom_left")
    print(bottom_left_search, "bottom_left_search")
    print(bottom_right, "bottom_right")
    print(bottom_right_search, "bottom_right_search")
    print(top_left_search_corner, "top_left_search_corner")

    return top_left, top_right, bottom_left, bottom_right, top_left_search, top_right_search, bottom_left_search, bottom_right_search, search_size, patchsize


def find_largest_2d_cluster(points, threshold=4):
    points = np.array(points)  # Convert to NumPy array
    print(f'points: {points}')

    clustering = DBSCAN(eps=threshold, min_samples=1, metric='euclidean').fit(points)
    print(f'clustering: {clustering}')
    # Organize points by cluster labels
    clusters = {}
    for point, label in zip(points, clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tuple(point))

    # Find the largest cluster
    largest_cluster = max(clusters.values(), key=len, default=[])

    return largest_cluster

# This should be at the top level of your script
def perform_single_match(args):
    """
    Worker function that now accepts an index and other debug arguments
    and passes them to the match function.
    """
    # --- UNPACK THE NEW, EXPANDED ARGUMENTS ---
    index, point_id, template, search_area, debug_flag, file2_path = args

    # Return a dictionary with an error status if the inputs are bad
    if 0 in search_area.shape or 0 in template.shape:
        return {'id': point_id, 'status': 'error', 'reason': 'Empty patch'}

    # This is your original brute-force method of generating all possible sub-patches
    patch = []
    locn = []
    for i in range(len(search_area) - len(template) + 1):
        for j in range(len(search_area[0]) - len(template[0]) + 1):
            patch.append(search_area[i:(i + len(template)), j:(j + len(template[0]))])
            locn.append((i, j))
            
    # --- CALL YOUR 'match' FUNCTION WITH ALL ARGUMENTS ---
    # The 'index', 'debug', and 'file2' arguments are now passed through.
    imgstab, locn_stab, confid = match(
        patch=patch, 
        template=template, 
        locn=locn, 
        search=search_area, 
        index=index, 
        debug=debug_flag, 
        file2=file2_path
    )
    
    # Return a dictionary containing results for logging
    return {
        'id': point_id,
        'status': 'success' if locn_stab is not None else 'error',
        'location': locn_stab,
        'template_shape': template.shape,
        'search_shape': search_area.shape,
        'reason': 'Match function failed' if locn_stab is None else None
    }


def find_match_refactored(file1, file2, scale_percent=100, shapefile_path=None, search_size_m=None, patch_size_m=None,replace_string="_aligned",debug=False):
    """
    Finds matches by reading full-resolution windows from disk, then resizing them
    according to scale_percent before matching. This avoids high memory usage while
    retaining the scaling logic.
    """

    start_time = time.time()
    print(f'start time: {start_time}')

    """
    A robust function that first checks if files exist, then checks if they are empty of data,
    before proceeding with the alignment.
    """
    # --- Step 1: Robustly check if the input files are valid ---

    # SIMPLE CHECK: Does the template file exist?
    if not os.path.exists(file1):
        raise FileNotFoundError(f"Template file does not exist: {file1}")

    # SIMPLE CHECK: Does the search file exist?
    if not os.path.exists(file2):
        raise FileNotFoundError(f"Search file does not exist: {file2}")


    if scale_percent <= 0 or scale_percent > 100:
        raise ValueError("scale_percent must be between 1 and 100.")

    # --- Step 1: Determine image properties and get metadata ---
    #multispectral = "altum" in file1.lower() or "_ms_" in file1.lower()
    #num_bands = 3 if multispectral else 4
    num_bands = 3

    with rasterio.open(file1) as src_template_meta:
        # No data is read here, just metadata
        pass

    with rasterio.open(file2) as src_search_meta:
        meta_search = src_search_meta.meta.copy()
        
        # This code peeks inside the file without using lots of memory.
        tags = src_search_meta.tags(1)
        max_val_tag = tags.get('STATISTICS_MAXIMUM')
        
        is_empty = False
        if max_val_tag is not None and float(max_val_tag) == 0:
            is_empty = True
        else:
            nodata_val = src_search_meta.nodata
            if nodata_val is not None:
                window = Window(0, 0, 10, 10) # Read a tiny 10x10 corner
                sample = src_search_meta.read(1, window=window)
                if np.all(sample == nodata_val):
                    is_empty = True

        if is_empty:
            raise IOError(f"Search file exists, but appears to be empty of data: {file2}")

    print("Info: Both files exist and contain data. Proceeding with alignment...")


    # --- Step 2: Get full-resolution points and dimensions from shapefile ---
    # This function must return pixel coordinates and sizes for the FULL-RESOLUTION images.

    geo_pts_with_ids, pts_template, pts_search, search_buffer_px, patch_size_px = get_patch_and_search_locations(
    shapefile_path=shapefile_path, 
    raster_file_for_coords=file1, 
    patch_sidelength_m=patch_size_m, 
    search_buffer_m=search_size_m)

    print(f"Full-res patch size: {patch_size_px}px, Full-res search buffer: {search_buffer_px}px")

    cp1 = time.time()
    print(f'Beginning template creation process, elapsed time so far: {cp1 - start_time}')

    # --- Step 3: Extract and Resize Template and Search windows ---
    templates_scaled = []
    print(f"\n--- Extracting and Resizing Template Patches (Scale: {scale_percent}%) ---")
    with rasterio.open(file1) as src:
        for i, pt in enumerate(pts_template):
            # 1. SLICE: Read the full-resolution window from disk
            template_window = Window(pt[1], pt[0], patch_size_px, patch_size_px)
            full_res_template_data = src.read(range(1, num_bands + 1), window=template_window)
            full_res_template_img = np.transpose(full_res_template_data, (1, 2, 0)).astype(np.float32)

            if 0 in full_res_template_img.shape:
                print(f"Warning: Skipped empty template patch at window {template_window}")
                continue

            # 2. RESIZE: Downscale the patch that is now in memory
            if scale_percent == 100:
                scaled_template_img = full_res_template_img
            else:
                width = int(full_res_template_img.shape[1] * scale_percent / 100)
                height = int(full_res_template_img.shape[0] * scale_percent / 100)
                scaled_template_img = cv.resize(full_res_template_img, (width, height), interpolation=cv.INTER_AREA)
            
            templates_scaled.append(scaled_template_img)
            print(f"Template {i+1}: Read {full_res_template_img.shape} -> Resized to {scaled_template_img.shape}")

        if debug:
            for i, template in enumerate(templates_scaled):
                debug_dir = Path(file2).parent / "debug"
                os.makedirs(debug_dir, exist_ok=True)
                template_bgr = cv.cvtColor(template, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(debug_dir, f"template_{i+1}.png"), template_bgr)


    cp2 = time.time()
    print(f'Beginning search area creation process, elapsed time so far: {cp2 - start_time}')


    searches_scaled = []
    print(f"\n--- Extracting and Resizing Search Areas (Scale: {scale_percent}%) ---")
    with rasterio.open(file2) as src:
        for i, pt in enumerate(pts_search):
            # 1. SLICE: Read the full-resolution search window from disk
            search_col = pt[1] - search_buffer_px
            search_row = pt[0] - search_buffer_px
            search_width = patch_size_px + (2 * search_buffer_px)
            search_height = patch_size_px + (2 * search_buffer_px)
            search_window = Window(search_col, search_row, search_width, search_height)
            
            full_res_search_data = src.read(range(1, num_bands + 1), window=search_window)
            full_res_search_img = np.transpose(full_res_search_data, (1, 2, 0)).astype(np.float32)

            if 0 in full_res_search_img.shape:
                print(f"Warning: Skipped empty search area at window {search_window}")
                continue

            # 2. RESIZE: Downscale the search area
            if scale_percent == 100:
                scaled_search_img = full_res_search_img
            else:
                width = int(full_res_search_img.shape[1] * scale_percent / 100)
                height = int(full_res_search_img.shape[0] * scale_percent / 100)
                scaled_search_img = cv.resize(full_res_search_img, (width, height), interpolation=cv.INTER_AREA)

            searches_scaled.append(scaled_search_img)
            print(f"Search Area {i+1}: Read {full_res_search_img.shape} -> Resized to {scaled_search_img.shape}")
 
        if debug:
            for i, search in enumerate(searches_scaled):
                debug_dir = Path(file2).parent / "debug"
                os.makedirs(debug_dir, exist_ok=True)
                search_bgr = cv.cvtColor(search, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(debug_dir, f"search_{i+1}.png"), search_bgr)

    cp3 = time.time()
    print(f'Begining matching process, elapsed time so far: {cp3 - start_time}')

    # --- Step 4: Perform template matching IN PARALLEL ---
    print("\n--- Matching Scaled Templates in Scaled Search Areas (in Parallel) ---")
    
    # 1. Prepare the list of jobs. Each job is a tuple of (template, search_area)
    # The zip() function creates an iterator of these pairs automatically.

    # --- NEW, INDEXED JOB PREPARATION ---
    jobs = []
    # Use enumerate to get a sequential index 'i' for each patch
    for i, (template, search_area) in enumerate(zip(templates_scaled, searches_scaled)):
        # Get the corresponding point ID from your shapefile data
        point_id = geo_pts_with_ids[i]['id']
        
        # Package everything the worker function will need into a single tuple:
        # (sequential_index, shapefile_id, template_data, search_data, debug_flag, file2_path)
        job_args = (i, point_id, template, search_area, debug, file2)
        jobs.append(job_args)
        results_from_workers = []
        num_processes = max(1, os.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        print(f"\n--- Matching templates in parallel using {num_processes} workers... ---")
        results_iterator = executor.map(perform_single_match, jobs)
        results_from_workers = list(results_iterator) # Ensure all jobs are finished


    # --- Step 5: Log Results and Collect Locations (NEW LOGGING SECTION) ---
    locns = []
    # Sort the results by point ID to guarantee a consistent print order
    results_from_workers.sort(key=lambda r: r.get('id', -1))
    
    # Print the header that was in your original log
    print("\n--- Matching Scaled Templates in Scaled Search Areas ---")
    
    for result in results_from_workers:
        # Check if the worker reported a success
        if result and result['status'] == 'success':
            # Recreate the exact output lines you wanted
            print(f"{result['search_shape']}")
            print(f"{result['template_shape']}")
            
            # This replicates the "best_match" line from your original log
            best_match_display = (result['template_shape'][0] // 2, result['template_shape'][1] // 2)
            print(f"best_match: {best_match_display}")
            
            # Print the final location found by the match function
            print(f"location: {result['location']}")
            
            # Collect the location for the next step (clustering)
            locns.append(result['location'])
        else:
            # Report any errors from the workers
            point_id = result.get('id', 'N/A')
            reason = result.get('reason', 'Unknown error')
            print(f"Skipped point ID {point_id} due to error: {reason}")

    
    print(f"Finished matching. Found {len(locns)} valid locations.")

    true_locns = find_largest_2d_cluster(points=locns, threshold=10)


    if not true_locns:
        print("Warning: No clusters found after matching. Cannot calculate transform.")
        return # Exit if no matches were found

    true_locns = np.array(true_locns)

    # Step 1: Calculate the search buffer in SCALED pixel units.
    scaled_search_buffer = int(search_buffer_px * scale_percent / 100)

    # Step 2: Calculate the average pixel shift in the SCALED coordinate system.
    # The Y-shift (dy) comes from the rows (index 0 of the coordinate tuple).
    scaled_delta_y = np.mean(true_locns[:, 0]) - scaled_search_buffer
    
    # The X-shift (dx) comes from the columns (index 1 of the coordinate tuple).
    scaled_delta_x = np.mean(true_locns[:, 1]) - scaled_search_buffer
    
    print(f"Debug: Scaled pixel shift (dx, dy) = ({scaled_delta_x:.2f}, {scaled_delta_y:.2f})")

    # Step 3: Scale the pixel shift back up to the FULL-RESOLUTION coordinate system.
    scaling_factor = 100 / scale_percent
    full_res_delta_x = scaled_delta_x * scaling_factor
    full_res_delta_y = scaled_delta_y * scaling_factor
    print(f"Debug: Full-resolution pixel shift (dx, dy) = ({full_res_delta_x:.2f}, {full_res_delta_y:.2f})")

    # Step 4: Convert the full-resolution pixel shift to a geographic shift in meters.
    pixel_width_m = meta_search['transform'].a  # How many meters the image moves per pixel shift in X
    pixel_height_m = meta_search['transform'].e # How many meters the image moves per pixel shift in Y (is negative)

    geo_delta_x = full_res_delta_x * pixel_width_m
    geo_delta_y = full_res_delta_y * pixel_height_m
    print(f"Debug: Geographic shift (dx, dy) = ({geo_delta_x:.4f}m, {geo_delta_y:.4f}m)")

    # Step 5: Apply the new transform to the file.
    # The geographic X shift (geo_delta_x) is added to the X-offset (c).
    # The geographic Y shift (geo_delta_y) is added to the Y-offset (f).
    
    # Rename file first to avoid read/write locks
    new_filepath = file2.replace(".tif", f"{replace_string}.tif")
    os.rename(file2, new_filepath)

    with rasterio.open(new_filepath, "r+") as dst:
        old_transform = dst.transform
        print(f"Old transform: {old_transform}")
        
        if "original_transform" not in dst.tags():
            dst.update_tags(original_transform=list(old_transform))
        dst.update_tags(aligned=True)

        new_transform = Affine(
            old_transform.a, old_transform.b, old_transform.c - geo_delta_x,
            old_transform.d, old_transform.e, old_transform.f - geo_delta_y
        )
        
        dst.transform = new_transform
        print(f"New transform: {new_transform}")
    
    print(f"File aligned and saved to: {new_filepath}")


def find_match(file1, file2, size=100, search_size=160, transform=np.float32([[1, 0, 0], [0, 1, 0]]),
               debug=False, output=False, matrix_path="", progress=False, save=True, lat_long_points=None,
               mark_and_save=False, shapefile_path=None, search_size_m=None, patch_size_m=None,
               top_right_idx=None, top_left_idx=None, bottom_left_idx=None, bottom_right_idx=None,
               replace_string="_aligned"):
    multispectral = False
    if "altum" in file1.lower() or "_ms_" in file1.lower():
        multispectral = True
        with rasterio.open(file1) as src:
            testimage = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1).astype(np.float32)

    else:

        with rasterio.open(file1) as src:
            testimage = np.stack([src.read(1), src.read(2), src.read(3), src.read(4)], axis=-1).astype(np.float32)
    with rasterio.open(file2) as src:
        meta = src.meta.copy()
    if mark_and_save:
        with rasterio.open(file1) as src:
            meta_template = src.meta.copy()
    if multispectral:
        with rasterio.open(file2) as src:
            search_original = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1).astype(np.float32)
            # search_original_other_bands = np.stack([src.read(4), src.read(5), src.read(6), src.read(7)], axis=-1).astype(np.float32)
    else:

        with rasterio.open(file2) as src:
            search_original = np.stack([src.read(1), src.read(2), src.read(3), src.read(4)], axis=-1).astype(np.float32)
    if np.max(search_original) > 0:
        pass
    else:
        print("no data in search image", file2)
        raise IOError

    scale_percent = size
    width = int(testimage.shape[1] * scale_percent / 100)
    height = int(testimage.shape[0] * scale_percent / 100)
    dim = (width, height)
    testimage_full_size = testimage
    testimage = cv.resize(testimage, dim, interpolation=cv.INTER_AREA)

    width = int(search_original.shape[1] * scale_percent / 100)
    height = int(search_original.shape[0] * scale_percent / 100)
    dim = (width, height)

    search_scaled = cv.resize(search_original, dim, interpolation=cv.INTER_AREA)

    if top_left_idx is None:
        top_left_idx = 0
    if top_right_idx is None:
        top_right_idx = 1
    if bottom_left_idx is None:
        bottom_left_idx = 2
    if bottom_right_idx is None:
        bottom_right_idx = 3

    next_avail_idx = max([top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx]) + 1

    """top_left, top_right, bottom_left, bottom_right, top_left_search, top_right_search, bottom_left_search, bottom_right_search, search_size, patchsize = return_pts_from_shapefile(
        shapefile_path, size, search_size_m, patch_size_m, file1, file2, top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx)
    """
    pts_template, pts_search, search_size, patchsize = return_all_pts_from_shapefile(shapefile_path, size,
                                                                                     search_size_m, patch_size_m, file1,
                                                                                     file2)

    if shapefile_path:
        pass
    else:
        search_size = int(search_size * scale_percent / 100)

    templates = []
    for pt in pts_template:
        template = testimage[pt[0]:(pt[0] + patchsize), pt[1]:(pt[1] + patchsize)]
        templates.append(template)
        print(pt, patchsize)
        print(template.shape, "shape of template")

    searches = []
    for pt in pts_search:
        print(pt, search_size)
        search = search_scaled[pt[0] - search_size:pt[0] + patchsize + search_size,
                 pt[1] - search_size:pt[1] + patchsize + search_size]
        searches.append(search)
        print(search.shape, "shape of search")

    """template1 = testimage[top_left[0]:(top_left[0] + patchsize), top_left[1]:(top_left[1] + patchsize)]
    template2 = testimage[top_right[0]:(top_right[0] + patchsize), top_right[1]:(top_right[1] + patchsize)]
    template3 = testimage[bottom_left[0]:(bottom_left[0] + patchsize), bottom_left[1]:(bottom_left[1] + patchsize)]
    template4 = testimage[bottom_right[0]:(bottom_right[0] + patchsize), bottom_right[1]:(bottom_right[1] + patchsize)]
    """

    locns = []
    for (template, search) in zip(templates, searches):
        patch = []
        locn = []
        print(search.shape)
        print(template.shape)
        if 0 in search.shape or 0 in template.shape:
            print("search or template is empty")
            continue

        for i in range(len(search) - len(template) + 1):
            for j in range(len(search[0]) - len(template[0]) + 1):
                patch.append(search[i:(i + len(template)), j:(j + len(template[0]))])
                locn.append((i, j))
        imgstab, locn_stab, confid = match(patch, template, locn, search)
        if locn_stab is not None:
            locns.append(locn_stab)
    print(locns, "all")
    true_locns = find_largest_2d_cluster(points=locns, threshold=10)
    print(true_locns, "cluster")
    anchor_pts = np.zeros((len(true_locns), 1, 2))

    new_pts = anchor_pts * 0

    i = 0

    """for locn in true_locns:
        anchor_pts[i][0][0] = pts_search[locns.index(locn)][0] + patchsize / 2
        anchor_pts[i][0][1] = pts_search[locns.index(locn)][1] + patchsize / 2
        new_pts[i][0][0] = pts_search[locns.index(locn)][0] - search_size + locn[0]
        new_pts[i][0][1] = pts_search[locns.index(locn)][1] - search_size + locn[1]

        i += 1
    print(anchor_pts, new_pts, "unscaled pts")

    anchor_pts_fullscale = anchor_pts * 100 / scale_percent
    new_pts_fullscale = new_pts * 100 / scale_percent

    print(anchor_pts_fullscale, new_pts_fullscale, "pts")


    m_full, ret_full = cv.estimateAffinePartial2D(new_pts_fullscale, anchor_pts_fullscale)

    new_transform = Affine(0, 0, m_full[0][2], 0, 0, m_full[1][2])
    print(new_transform)"""

    true_locns = np.array(true_locns)
    delta_x = np.mean(true_locns[:, 0]) - search_size
    delta_y = np.mean(true_locns[:, 1]) - search_size
    if multispectral:

        delta_x *= 100 / size * .03
        delta_y *= 100 / size * .03  # adjusting for altum gsd
    else:
        delta_x *= 100 / size * 0.007
        delta_y *= 100 / size * 0.007
    with rasterio.open(file2, "r+") as dst:
        old_trans_printable = [dst.transform.a, dst.transform.b, dst.transform.c, dst.transform.d, dst.transform.e,
                               dst.transform.f]
        print(old_trans_printable)
        if "original_transform" in dst.tags():
            pass
        else:
            dst.update_tags(original_transform=old_trans_printable)
        dst.update_tags(aligned=True)
        new_transform = [dst.transform.a, dst.transform.b, dst.transform.c - delta_y, dst.transform.d, dst.transform.e,
                         dst.transform.f + delta_x]
        dst.transform = Affine(new_transform[0], new_transform[1], new_transform[2], new_transform[3], new_transform[4],
                               new_transform[5])
    os.rename(file2, file2.replace(".tif", f"{replace_string}.tif"))


def run_entire_set_ortho(filepath, start_file):
    list_of_file_names = os.listdir(filepath)
    list_of_file_names = [_ for _ in list_of_file_names if _.lower().endswith(".tif")]
    list_of_file_names = sorted(list_of_file_names)
    mid_index = list_of_file_names.index(start_file.split("/")[-1])
    list_of_file_names_front = list_of_file_names[:mid_index]
    list_of_file_names_back = list_of_file_names[mid_index + 1:]
    list_of_file_names_front.reverse()

    # shutil.copy(start_file, start_file.replace("raw", "adjusted_utm"))

    first_filename = start_file.split("/")[-1]

    for file_name in list_of_file_names_back:
        print(filepath.replace("raw", "adjusted_utm") + first_filename, filepath + file_name)
        find_match(filepath.replace("raw", "adjusted_utm") + first_filename, filepath + file_name,
                   shapefile_path="C:/Users/Ryan Waltz/OneDrive - The Ohio State University/nitrogen/uas_pipeline/orthomosaic_correction/western_bftb_pts/western_bftb_pts.shp",
                   search_size_m=15 * .4, patch_size_m=7.5 * .8, size=25, progress=True,
                   output=filepath.replace("raw", "adjusted_utm"))
        first_filename = file_name

    # 516 western sony is back
    # i need to do what I've already done in the past with the image stabilization - I need to have a matrix of transformation that I premanipulate the image so it doesn't have to move so much
    first_filename = start_file.split("/")[-1]
    for file_name in list_of_file_names_front:
        find_match(filepath.replace("raw", "adjusted_utm") + first_filename, filepath + file_name,
                   shapefile_path="C:/Users/Ryan Waltz/OneDrive - The Ohio State University/nitrogen/uas_pipeline/orthomosaic_correction/western_bftb_pts/western_bftb_pts.shp",
                   search_size_m=15 * .4, patch_size_m=7.5 * .8, size=25, progress=True,
                   output=filepath.replace("raw", "adjusted_utm"))
        first_filename = file_name


def run_list_of_files(forward_pass, backward_pass, first_filename, shapefile_path, replace_string="_aligned"):
    # shutil.copy(first_filename, first_filename.replace("orthomosaics", "orthomosaics_aligned_rw"))
    true_first = first_filename
    os.rename(first_filename, first_filename.replace(".tif", "_aligned.tif"))
    if "_ms_" in first_filename or "altum" in first_filename:
        size = 25
    else:
        size = 8

    for filename in backward_pass:
        print(filename, first_filename)
        try:
            find_match(first_filename.replace(".tif", "_aligned.tif"), filename,
                       shapefile_path=shapefile_path,
                       search_size_m=6, patch_size_m=6, size=size, progress=True, replace_string=replace_string)

            first_filename = filename
        except rasterio.errors.RasterioIOError as e:
            print("rasterio error: ", e)
        except IOError as e:
            print("this image has no data")

    first_filename = true_first
    for filename in forward_pass:
        print(filename, first_filename)
        try:
            find_match(first_filename.replace(".tif", "_aligned.tif"), filename,
                       shapefile_path=shapefile_path,
                       search_size_m=6, patch_size_m=6, size=size, progress=True, replace_string=replace_string)
            first_filename = filename
        except rasterio.errors.RasterioIOError as e:
            print("rasterio error: ", e)
        except IOError as e:
            print("this image has no data")


def run_list_of_files_fine_tuned(forward_pass, backward_pass, first_filename, shapefile_path,
                                 replace_string="_finetuned"):
    # shutil.copy(first_filename, first_filename.replace("orthomosaics", "orthomosaics_aligned_rw"))
    true_first = first_filename
    os.rename(first_filename, first_filename.replace(".tif", f"{replace_string}.tif"))

    for filename in backward_pass:
        print(filename, first_filename)
        try:
            find_match(first_filename.replace(".tif", "_finetuned.tif"), filename,
                       shapefile_path=shapefile_path,
                       search_size_m=0.5, patch_size_m=6, size=100, progress=True, replace_string=replace_string)

            first_filename = filename
        except rasterio.errors.RasterioIOError as e:
            print("rasterio error: ", e)
        except IOError as e:
            print("this image has no data")

    first_filename = true_first
    for filename in forward_pass:
        print(filename, first_filename)
        try:
            find_match(first_filename.replace(".tif", "_finetuned.tif"), filename,
                       shapefile_path=shapefile_path,
                       search_size_m=0.5, patch_size_m=6, size=100, progress=True, replace_string=replace_string)
            first_filename = filename
        except rasterio.errors.RasterioIOError as e:
            print("rasterio error: ", e)
        except IOError as e:
            print("this image has no data")


def find_template(pattern, overall_folder, shapefile_path, file_to_align, split_pattern=None):
    potentials = os.listdir(overall_folder)
    desired = []
    for i in potentials:
        if pattern in i.lower() and i.endswith(".tif") and (not "incomplete" in i):
            desired.append(overall_folder + i)

    desired = sorted(desired)
    print(f'desired: {desired}')
    desired = desired[:desired.index(overall_folder + file_to_align) + 1]  # shouldn't be necessary, but just in case

    i = -2
    while True:
        try:
            if "aligned" in desired[i]:
                template = desired[i]
                break
            i -= 1
        except IndexError:
            return None, None
    shapefile_path = shapefile_path + pattern.replace("_altum", "").replace("_sony", "").replace("_ms", "").replace(
        "_rgb", "") + "_pts/" + pattern.replace("_altum", "").replace("_sony", "").replace("_ms", "").replace("_rgb",
                                                                                                              "") + "_pts.shp"
    return template, shapefile_path

if __name__ == "__main__":
    # Essentially, we will have two (possibly parallel) processes. One will be the forward facing pass, and the other
    # will be the backward facing pass. The forward facing pass will take in the middle image and propogate forwards,
    # whereas the backward facing pass will take in the middle image and propogate backwards. Both will use the same features,
    # which will likely be established by geographic coordinates. The images are massive, so this creates certain challenges
    # if we want to use this economically.

    parser = argparse.ArgumentParser(description="Run orthomosaic correction")
    parser.add_argument("--pattern", type=str, required=True, help="template")
    parser.add_argument("--search", type=str, required=True, help="search")
    parser.add_argument("--om_folder", type=str, required=True, help="Path to om")
    parser.add_argument("--om_aligned_folder", type=str, required=True, help="Path to om")
    parser.add_argument("--shapefile_path", type=str, required=True, help="Path to shapefile")
    parser.add_argument("--fine_tuned", type=str, required=False, default="n",
                        help="Only for use with prealigned images")

    args = parser.parse_args()

    print(f'pattern: {args.pattern}')
    print(f'search: {args.search}')
    print(f'om folder: {args.om_folder}')
    print(f'om aligned folder: {args.om_aligned_folder}')
    print(f'shapefile folder: {args.shapefile_path}')
    print(f'fine tuned: {args.fine_tuned}')



    shutil.copy(os.path.join(args.om_folder, args.search), os.path.join(args.om_aligned_folder, args.search))
    template, shapefile_path = find_template(args.pattern, args.om_aligned_folder, args.shapefile_path, args.search)

    print(f'template: {template}')
    print(f'shapefile path: {shapefile_path}')


    if template is None:
        shutil.move(os.path.join(args.om_aligned_folder, args.search), os.path.join(args.om_aligned_folder, args.search).replace(".tif", "_aligned.tif"))
        exit()
    if "ms" in template:
        scale_percent = 25
    else:
        scale_percent = 8
    print(f'template: {template}, search: {args.search}, shapefile_path: {shapefile_path}')

    start_time = time.time()
    """find_match(template, args.om_aligned_folder+ args.search, shapefile_path=shapefile_path, search_size_m=6, patch_size_m=6,
               replace_string="_aligned", size=size)"""
    
    search_image = os.path.join(args.om_aligned_folder, args.search)
    find_match_refactored(template, search_image, shapefile_path=shapefile_path, search_size_m=6, patch_size_m=6,
               replace_string="_aligned", scale_percent=scale_percent, debug=True)
    