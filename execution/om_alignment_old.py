import os
import pathlib
import shutil

import cv2
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
import shapefile
import argparse
from pathlib import Path

from sklearn.cluster import DBSCAN

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


def match(patch, template, locn, search, index, debug = False, file2 = None):
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
        debug_dir = Path(file2).parent / "debug_old"
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
            for shape in shp.shapes():
                pts.append(shape.points[0])
                print(f'point: {shape.points[0]}')

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


def find_match(file1, file2, size=100, search_size=160, transform=np.float32([[1, 0, 0], [0, 1, 0]]),
               output=False, matrix_path="", progress=False, save=True, lat_long_points=None,
               mark_and_save=False, shapefile_path=None, search_size_m=None, patch_size_m=None,
               top_right_idx=None, top_left_idx=None, bottom_left_idx=None, bottom_right_idx=None,
               replace_string="_aligned", debug = False):
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

    print(f'pts_template: {pts_template}')
    print(f'pts_search: {pts_search}')
    print(f'search_size: {search_size}')
    print(f'patchsize: {patchsize}')


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
    if debug:
        for i, template in enumerate(templates):
            if template is not None and template.size > 0:
                debug_dir = Path(file2).parent / 'debug_old'
                os.makedirs(debug_dir, exist_ok=True)
                template_bgr = cv.cvtColor(template, cv.COLOR_BGR2RGB)
                cv.imwrite(debug_dir / f'template_{i+1}.png', template_bgr)
            else:
                print(f"Warning: Skipping image at index {i} because it failed to load.")


    searches = []
    for pt in pts_search:
        print(pt, search_size)
        search = search_scaled[pt[0] - search_size:pt[0] + patchsize + search_size,
                 pt[1] - search_size:pt[1] + patchsize + search_size]
        searches.append(search)
        print(search.shape, "shape of search")

    if debug:
        for i, search in enumerate(searches):
            if search is not None and search.size > 0:
                print(f'search shape: {search.shape}')
                debug_dir = Path(file2).parent / 'debug_old'
                os.makedirs(debug_dir, exist_ok=True)
                search_bgr = cv.cvtColor(search, cv.COLOR_BGR2RGB)
                cv.imwrite(debug_dir / f'search_{i}.png', search_bgr)
            else:
                print(f"Warning: Skipping image at index {i} because it failed to load.")


    """template1 = testimage[top_left[0]:(top_left[0] + patchsize), top_left[1]:(top_left[1] + patchsize)]
    template2 = testimage[top_right[0]:(top_right[0] + patchsize), top_right[1]:(top_right[1] + patchsize)]
    template3 = testimage[bottom_left[0]:(bottom_left[0] + patchsize), bottom_left[1]:(bottom_left[1] + patchsize)]
    template4 = testimage[bottom_right[0]:(bottom_right[0] + patchsize), bottom_right[1]:(bottom_right[1] + patchsize)]
    """

    locns = []
    for k, (template, search) in enumerate(zip(templates, searches)):
        patch = []
        locn = []
        print(f'index: {k}')
        print(f'search.shape: {search.shape}')
        print(f'template.shape: {template.shape}')
        if 0 in search.shape or 0 in template.shape:
            print("search or template is empty")
            continue

        for i in range(len(search) - len(template) + 1):
            for j in range(len(search[0]) - len(template[0]) + 1):
                patch.append(search[i:(i + len(template)), j:(j + len(template[0]))])
                locn.append((i, j))
        imgstab, locn_stab, confid = match(patch, template, locn, search, k, debug, file2)
        if locn_stab is not None:
            locns.append(locn_stab)
    print(f'Locations: {locns}')
    true_locns = find_largest_2d_cluster(points=locns, threshold=10)
    print(f'True locations: {true_locns}')
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
        print(f'Old transform: {old_trans_printable}')
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

    os.makedirs(args.om_aligned_folder, exist_ok=True)


    shutil.copy(os.path.join(args.om_folder, args.search), os.path.join(args.om_aligned_folder, args.search))
    template, shapefile_path = find_template(args.pattern, args.om_aligned_folder, args.shapefile_path, args.search)

    print(f'template: {template}')
    print(f'shapefile path: {shapefile_path}')


    if template is None:
        shutil.move(os.path.join(args.om_aligned_folder, args.search), os.path.join(args.om_aligned_folder, args.search).replace(".tif", "_aligned.tif"))
        exit()
    if "ms" in template:
        size = 25
    else:
        size = 8
    print(f'template: {template}, search: {args.search}, shapefile_path: {shapefile_path}')
    find_match(template, args.om_aligned_folder+ args.search, shapefile_path=shapefile_path, search_size_m=6, patch_size_m=6,
               replace_string="_aligned", size=size, debug=False)