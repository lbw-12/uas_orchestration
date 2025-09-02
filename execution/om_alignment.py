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

class ConfidenceError(Exception):
    def __init__(self, value):
        self.value = value

def match(patch, template, locn, search, index,debug=False, file2=None):
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
        debug_dir = Path(file2).parent / "debug_new"
        os.makedirs(debug_dir, exist_ok=True)
        best_match = patch[int(ncc_sorted[0][0])]
        best_match_center = (best_match.shape[0] // 2, best_match.shape[1] // 2)
        best_match_annotated = best_match.copy()
        # Get the shape of the image array
        height, width, num_channels = best_match_annotated.shape

        # Define the coordinates for clarity
        y, x = best_match_center[0], best_match_center[1]

        # Check the number of channels and assign the appropriate color
        if num_channels == 4:
            # It's an RGBA or BGRA image, so provide 4 values
            best_match_annotated[y, x] = [0, 0, 255, 255]  # Red, with full opacity
        else:
            # It's an RGB or BGR image, so provide 3 values
            best_match_annotated[y, x] = [0, 0, 255]

        # Annotate the template with a green dot of the highest rated patch location
        #template_center = (locn[int(ncc_sorted[0][0])][0] + best_match.shape[0] // 2, locn[int(ncc_sorted[0][0])][1] + best_match.shape[1] // 2)
        template_annotated = template.copy()
        #template_annotated[template_center[0], template_center[1]] = [0, 0, 255]
        template_center = (template.shape[0] // 2, template.shape[1] // 2)

        y, x = template_center[0], template_center[1]
        if num_channels == 4:
            template_annotated[y, x] = [0, 0, 255, 255]
        else:
            template_annotated[y, x] = [0, 0, 255]

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

def return_all_pts_from_shapefile(shapefile_path, size, search_size_m, patch_size_m, file1, file2):
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
            new_pt = [pt[0] - patch_size_m / 2, pt[1] - patch_size_m]
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

    return pts_template_pxl[:-1], pts_search_pxl[:-1], search_size, patchsize

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
    multispectral = "altum" in file1.lower() or "_ms_" in file1.lower()
    num_bands = 3 if multispectral else 4

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
    pts_template, pts_search, search_buffer_px, patch_size_px = return_all_pts_from_shapefile(
        shapefile_path, 100, search_size_m, patch_size_m, file1, file2
    )
    print(f"Full-res patch size: {patch_size_px}px, Full-res search buffer: {search_buffer_px}px")
    print(f'pts_template: {pts_template}')
    print(f'pts_search: {pts_search}')
    print(f'search_size: {search_buffer_px}')
    print(f'patch_size_px: {patch_size_px}')

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
                debug_dir = Path(file2).parent / "debug_new"
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
                print(f"Warning: Found empty search area at window {search_window}")
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
                debug_dir = Path(file2).parent / "debug_new"
                os.makedirs(debug_dir, exist_ok=True)
                search_bgr = cv.cvtColor(search, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(debug_dir, f"search_{i+1}.png"), search_bgr)

    cp3 = time.time()
    print(f'Begining matching process, elapsed time so far: {cp3 - start_time}')

    # --- Step 4: Perform template matching on the scaled-down images ---
    print("\n--- Matching Scaled Templates in Scaled Search Areas ---")
    locns = []
    for k, (template, search) in enumerate(zip(templates_scaled, searches_scaled)):
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
        imgstab, locn_stab, confid = match(patch, template, locn, search, k, debug=debug, file2=file2)
        if locn_stab is not None:
            locns.append(locn_stab)
    print(locns, "all")
    true_locns = find_largest_2d_cluster(points=locns, threshold=10)
    print(true_locns, "cluster")
    anchor_pts = np.zeros((len(true_locns), 1, 2))

    new_pts = anchor_pts * 0
    i = 0

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
    
    search_image = os.path.join(args.om_aligned_folder, args.search)
    find_match_refactored(template, search_image, shapefile_path=shapefile_path, search_size_m=6, patch_size_m=6,
               replace_string="_aligned", scale_percent=scale_percent, debug=True)
    