import os
import numpy as np
import json
from skimage import io
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import joblib
from colorspacious import cspace_convert
import argparse

def load_model(model_path):
    return joblib.load(model_path)

def classify_image(image_path, kmeans):
    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Discard alpha if present

    lab = rgb2lab(image)
    nodata_mask = np.all(lab == [0, 128, 128], axis=-1) | np.all(lab == [0, 0, 0], axis=-1)

    lab_ab = lab[:, :, 1:3].reshape(-1, 2)
    labels = kmeans.predict(lab_ab).reshape(lab.shape[:2])

    centroids = kmeans.cluster_centers_
    green_cluster, brown_cluster = sorted(range(2), key=lambda i: centroids[i][0])

    classification = np.full(lab.shape[:2], '', dtype='<U10')
    classification[labels == green_cluster] = 'green'
    classification[labels == brown_cluster] = 'brown'
    classification[nodata_mask] = 'nodata'

    return classification

def calculate_green_percentage(classification):
    green_pixels = np.sum(classification == 'green')
    valid_pixels = np.sum(np.logical_or(classification == 'green', classification == 'brown'))
    return (green_pixels / (valid_pixels + 1e-6))

def process_folder(input_dir, model_path, output_json_path, file_ext=".tif"):
    kmeans = load_model(model_path)
    results = {}

    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(file_ext) and not fname.startswith("._"):
            image_path = os.path.join(input_dir, fname)
            try:
                classification = classify_image(image_path, kmeans)
                canopy_percentage = calculate_green_percentage(classification)
                results[fname] = canopy_percentage
                print(f"{fname}: {canopy_percentage:.2f}%")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                results[fname] = None

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved canopy coverage results to: {output_json_path}")

if __name__ == "__main__":
    #get agrs
    parser = argparse.ArgumentParser(description="Process canopy cover images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained MiniBatchKMeans model.")
    
    args = parser.parse_args()
    year_list = ['2023', '2024', '2025']
    # get year from input_dir
    input_dir = args.input_dir
    year = None
    for y in year_list:
        if y in input_dir:
            year = y
            break
    process_folder(args.input_dir, args.model_path, args.output_json)