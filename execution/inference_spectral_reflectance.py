import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import cv2
import pdb
import json
import argparse
import joblib
from collections import defaultdict

def analyze_image_statistics(image, mask, ndvi, image_name, stats_list):
    veg_pixels = image[mask == 1]
    if veg_pixels.size == 0:
        median_bands = [None] * image.shape[2]
    else:
        median_bands = np.median(veg_pixels, axis=0).tolist()

    ndvi_mean = float(np.nanmean(ndvi))
    vegetation_cover = float(np.sum(mask) / mask.size)

    stats_list.append({
        "image": image_name,
        "median_spectral_bands": median_bands,
        "mean_ndvi": ndvi_mean,
        "vegetation_cover": vegetation_cover
    })

def save_metrics_to_json(results, output_json_path, field, plotimage_source, date):

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
            print(f"Found and stored {len(ordered_plots_dict)} spectral reflectance plots for '{crop}'.")
        else:
            print(f"Warning: No spectral reflectance results found for '{crop}'. Skipping.")

    # Check if the final results dictionary contains any crop data before writing.
    if final_results[field]:
        print(f"Structuring results for {len(final_results[field])} crop(s) with data.")

        # Save the final structured dictionary to JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"✅ Combined spectral reflectance results saved to: {output_json_path}")
    else:
        # If final_results[field] is empty, it means no crops had valid data.
        print("Error: No valid spectral reflectance predictions found. JSON file will not be created.")



def save_side_by_side_overlay(image, mask, out_path, alpha=0.4):
    rgb = image[..., [2, 1, 0]]
    rgb_norm = rgb / np.percentile(rgb, 99)
    rgb_norm = np.clip(rgb_norm, 0, 1)
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

    overlay = np.zeros_like(rgb_uint8)
    overlay[mask == 1] = [255, 255, 255]

    blended = cv2.addWeighted(rgb_uint8, 1 - alpha, overlay, alpha, 0)

    # Side-by-side
    side_by_side = np.concatenate([rgb_uint8, blended], axis=1)
    cv2.imwrite(out_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

def save_stacked_overlay_with_stats(image, mask, ndvi, stats, out_path, alpha=0.5):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from matplotlib.gridspec import GridSpec

    # Normalize RGB
    rgb = image[..., [2, 1, 0]]
    rgb_norm = rgb / np.percentile(rgb, 99)
    rgb_norm = np.clip(rgb_norm, 0, 1)
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

    # Create white overlay
    overlay = np.zeros_like(rgb_uint8)
    overlay[mask == 1] = [255, 255, 255]
    blended = cv2.addWeighted(rgb_uint8, 1 - alpha, overlay, alpha, 0)

    # Set up plot
    fig = plt.figure(figsize=(6, 6.8))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.2])
    gs.update(hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Top: RGB
    ax1.imshow(rgb_uint8)
    ax1.set_title("Original RGB Image", fontsize=11)
    ax1.axis("off")

    # Middle: Overlay
    ax2.imshow(rgb_uint8)
    ax2.imshow(overlay, alpha=alpha)
    ax2.set_title("Vegetation Segmentation (White Overlay)", fontsize=11)
    ax2.axis("off")

    # Bottom: Stats (centered)
    ax3.axis("off")
    band_labels = ["Blue", "Green", "Red", "Red Edge", "NIR", "Thermal"]
    bands = stats.get("median_spectral_bands", [])

    if bands and None not in bands:
        band_string = "  ".join([f"{label}={val:.3f}" for label, val in zip(band_labels, bands)])
    else:
        band_string = "Median Reflectance: N/A"

    stats_text = (
        f"{band_string}\n"
        f"Mean NDVI: {stats['mean_ndvi']:.3f}    |    Vegetation Cover: {stats['vegetation_cover'] * 100:.2f}%"
    )
    ax3.text(0.5, 0.5, stats_text, fontsize=10, ha='center', va='center')

    # Save with zero whitespace
    plt.subplots_adjust(top=0.96, bottom=0.04, left=0.05, right=0.95)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def load_multiband_tif(path):
    with rasterio.open(path) as src:
        img = src.read(list(range(1, 7))).astype(np.float32)  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, 6)
    return img

def calculate_ndvi(image):
    R = image[..., 2]
    NIR = image[..., 4]
    return (NIR - R) / (NIR + R + 1e-5)

def save_geotiff(reference_path, output_path, array, dtype='float32'):
    with rasterio.open(reference_path) as src:
        profile = src.profile
        profile.update(dtype=dtype, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(array.astype(dtype), 1)

def save_mask_as_png(mask, image, out_path, alpha=0.4):
    """
    Overlays the binary vegetation mask on the RGB image and saves it as PNG.
    """
    rgb = image[..., [2, 1, 0]]  # Convert to RGB from BGR order
    rgb = rgb / np.percentile(rgb, 99)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    overlay = np.zeros_like(rgb)
    overlay[mask == 1] = [255, 255, 255]  # white vegetation overlay

    blended = cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

def predict_and_save(files, output_dir, clf, field, plotimage_source, date, visualize=False, output_json = "inference_spectral_reflectance.json"):

    
    stats_list = []
    mean_ndvi = {}
    for path in tqdm(files, desc="Predicting and saving"):
        image = load_multiband_tif(path)
        ndvi = calculate_ndvi(image)
        ndvi_mask = ndvi > 0.45

        H, W, _ = image.shape
        flat_features = image.reshape(-1, 6)

        pred_mask = clf.predict(flat_features).reshape(H, W)
        base = os.path.basename(path).replace(".tif", "")
        analyze_image_statistics(image,pred_mask, ndvi, base, stats_list)
        stats = stats_list[-1]
        mean_ndvi[os.path.basename(path)] = float(np.nanmean(ndvi))
        print(f"{base}: Mean NDVI = {mean_ndvi[os.path.basename(path)]:.4f}, Vegetation Cover = {stats['vegetation_cover'] * 100:.2f}%")
        if visualize:
            os.makedirs(output_dir, exist_ok=True)
            
            out_mask_path = os.path.join(output_dir, f"{base}_mlmask.tif")
            out_png_path = os.path.join(output_dir, f"{base}_mlmask.png")
            sidebyside_png_path = os.path.join(output_dir, f"{base}_ndvi_sidebyside.png")
            overlay_png_path = os.path.join("ml_outputs", f"{base}_ndvi_overlay.png")
            save_mask_as_png(ndvi_mask, image, overlay_png_path)
            save_geotiff(path, out_mask_path, pred_mask.astype("uint8"))
            save_mask_as_png(pred_mask, image, out_png_path)
            save_side_by_side_overlay(image, pred_mask, sidebyside_png_path)
            save_stacked_overlay_with_stats(image, pred_mask, ndvi, stats, sidebyside_png_path)
          
    save_metrics_to_json(mean_ndvi, output_json, field, plotimage_source, date)

def get_ndvi_json(image_dir, ndvi_json_path):
    """
    Generates a JSON file with NDVI values for each image in the directory.
    """
    ndvi_dict = {}
    all_files = sorted(glob(os.path.join(image_dir, "*.tif")))

    for path in tqdm(all_files, desc="Calculating NDVI"):
        image = load_multiband_tif(path)
        ndvi = calculate_ndvi(image)
        mean_ndvi = np.nanmean(ndvi)
        base = os.path.basename(path)
        ndvi_dict[base] = float(mean_ndvi)
        print(f"{base}: {mean_ndvi:.4f}")

    with open(ndvi_json_path, "w") as f:
        json.dump(ndvi_dict, f, indent=2)
    #close the file
    f.close()
    print(f"NDVI values saved to {ndvi_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Run UAS Multispectral Inference Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing multispectral images")
    parser.add_argument("--output_dir", type=str, default="ml_outputs", help="Directory to save output masks and metrics")
    parser.add_argument("--output_json", type=str, default="inference_spectral_reflectance.json", help="Path to save the output JSON file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pre-trained model (if any)")
    parser.add_argument("--visualize", type=bool, default=False, help="Whether to visualize results with overlays and side-by-side images")
    parser.add_argument('--field', type=str, default=None, help='Field ID')
    parser.add_argument('--plotimage_source', type=str, default=None, help='Plot image source')
    parser.add_argument('--date', type=str, default=None, help='Date')

    args = parser.parse_args()

    all_files = sorted(glob(os.path.join(args.input_dir, "*.tif")))
    #load the model 
    clf = joblib.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

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

    predict_and_save(all_files, args.output_dir, clf, field, plotimage_source, date, visualize=args.visualize, output_json=args.output_json)
   
if __name__ == "__main__":
    main()