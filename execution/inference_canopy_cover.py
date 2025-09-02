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
from collections import defaultdict

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

def process_folder(input_dir, model_path, output_json_path, field, plotimage_source, date, file_ext=".tif"):
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

    # 1. Group predictions by crop type first.
    # We use a defaultdict to easily create new crop lists as we find them.
    predictions_by_crop = defaultdict(dict)
    print("Validating plot numbers and grouping by crop...")

    for image_name, pred in results.items():
       # --- A. Extract Crop Name ---
        crop_name = None
        if '_corn_' in image_name:
            crop_name = 'corn'
        elif '_soy_' in image_name:
            crop_name = 'soy'
        elif '_cc_' in image_name:
            crop_name = 'cc'
        elif '_sc_' in image_name:
            crop_name = 'sc'

        # --- B. Extract and Validate Plot Number ---
        plot_str = image_name.split('_')[-2]

        # --- C. Add to the correct group if valid ---
        if crop_name and plot_str.isdigit() and len(plot_str) == 3:
            # Add the plot and its prediction to the correct crop's dictionary
            predictions_by_crop[crop_name][plot_str] = round(float(pred), 4)
        else:
            # If crop or plot format is invalid, print a warning.
            print(f"  [!] Warning: Skipping invalid format in filename '{image_name}'.")

     # 2. Sort the plots within each crop group and build the final structure.
    final_results = {field: {}} # Start with the top-level field key

    for crop, plots_dict in predictions_by_crop.items():
        print(f"Sorting {len(plots_dict)} plots for crop: '{crop}'")

        # Sort the plots for the current crop numerically
        sorted_plots = sorted(plots_dict.items(), key=lambda item: int(item[0]))
        ordered_plots_dict = dict(sorted_plots)

        if ordered_plots_dict:
            # Build the nested structure for this specific crop
            final_results[field][crop] = {
                plotimage_source: {
                    date: ordered_plots_dict
                }
            }
            print(f"Found and stored {len(ordered_plots_dict)} canopy coverage plots for '{crop}'.")
        else:
            print(f"Warning: No canopy coverage results found for '{crop}'. Skipping.")

    # Save the final structured dictionary to JSON
    if final_results[field]:
        print(f"Structuring results for {len(final_results[field])} crop(s) with data.")

        # Save the final structured dictionary to JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"✅ Combined canopy coverage results saved to: {output_json_path}")
    else:
        # If final_results[field] is empty, it means no crops had valid data.
        print("Error: No valid canopy coverage predictions found. JSON file will not be created.")
        
if __name__ == "__main__":
    #get agrs
    parser = argparse.ArgumentParser(description="Process canopy cover images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained MiniBatchKMeans model.")
    parser.add_argument('--field', type=str, default=None, help='Field ID')
    parser.add_argument('--plotimage_source', type=str, default=None, help='Plot image source')
    parser.add_argument('--date', type=str, default=None, help='Date')
 
    
    args = parser.parse_args()
    year_list = ['2023', '2024', '2025']
    # get year from input_dir
    input_dir = args.input_dir
    year = None
    for y in year_list:
        if y in input_dir:
            year = y
            break

    field = args.field
    plotimage_source = args.plotimage_source
    date = args.date

    process_folder(args.input_dir, args.model_path, args.output_json, field, plotimage_source, date)