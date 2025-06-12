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

def save_metrics_to_json(stats_list, out_path="vegetation_metrics.json"):
    with open(out_path, "w") as f:
        json.dump(stats_list, f, indent=2)

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

def extract_features_and_labels(files, threshold=0.35):
    feature_list = []
    label_list = []

    for path in tqdm(files, desc="Extracting features"):
        image = load_multiband_tif(path)
        ndvi = calculate_ndvi(image)
        mask = ndvi > threshold

        #base = os.path.basename(path).replace(".tif", "")
        #overlay_png_path = os.path.join("ml_outputs", f"{base}_ndvi_overlay.png")
        #save_mask_as_png(mask, image, overlay_png_path)

        H, W, _ = image.shape
        #features = np.concatenate([image, ndvi[..., None]], axis=-1)
        #features = features.reshape(-1, 7)
        features = image.reshape(-1, 6)
        labels = mask.astype(int).flatten()

        feature_list.append(features)
        label_list.append(labels)

    all_features = np.vstack(feature_list)
    all_labels = np.hstack(label_list)
    return all_features, all_labels

def train_classifier(features, labels):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, stratify=labels)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Validation Report:\n", classification_report(y_val, y_pred))
    return clf

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

def predict_and_save(files, clf, output_dir="ml_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    stats_list = []
    for path in tqdm(files, desc="Predicting and saving"):
        image = load_multiband_tif(path)
        ndvi = calculate_ndvi(image)
        ndvi_mask = ndvi > 0.45

        H, W, _ = image.shape
        #features = np.concatenate([image, ndvi[..., None]], axis=-1)
        #flat_features = features.reshape(-1, 7)
        flat_features = image.reshape(-1, 6)

        pred_mask = clf.predict(flat_features).reshape(H, W)

        base = os.path.basename(path).replace(".tif", "")
        out_mask_path = os.path.join(output_dir, f"{base}_mlmask.tif")
        out_png_path = os.path.join(output_dir, f"{base}_mlmask.png")
        sidebyside_png_path = os.path.join(output_dir, f"{base}_ndvi_sidebyside.png")
        overlay_png_path = os.path.join("ml_outputs", f"{base}_ndvi_overlay.png")
        save_mask_as_png(ndvi_mask, image, overlay_png_path)
        save_geotiff(path, out_mask_path, pred_mask.astype("uint8"))
        save_mask_as_png(pred_mask, image, out_png_path)
        save_side_by_side_overlay(image, pred_mask, sidebyside_png_path)

        analyze_image_statistics(image,pred_mask, ndvi, base, stats_list)
        stats = stats_list[-1]  # Get latest stats
        save_stacked_overlay_with_stats(image, pred_mask, ndvi, stats, sidebyside_png_path)
        report = classification_report(pred_mask.flatten(), ndvi_mask.flatten(), output_dict=True)

        if '1' in report:
            print(f"F1 score: {report['1']['f1-score']:.4f}")
        else:
            print("F1 score: N/A (no vegetation predicted or present in ground truth)")
                
    save_metrics_to_json(stats_list, out_path=os.path.join(output_dir, "vegetation_metrics.json"))

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
    pdb.set_trace()
    image_dir = "/fs/ess/PAS2699/nitrogen/data/uas/2024/plottiles/plot_tiles_multispectral_om"  # <<< UPDATE THIS
    threshold = 0.45
    output_dir = "ml_outputs"
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(glob(os.path.join(image_dir, "*.tif")))
    subset_train_files = all_files[:200]
    subset_test_files = all_files[:300]
    
    print("Step 0: Generating NDVI JSON")
    ndvi_json_path = os.path.join(output_dir, "ndvi_values.json")
    get_ndvi_json(image_dir, ndvi_json_path)
    
    print("Step 1: Extracting Features and Labels from 500 Training Images")
    features, labels = extract_features_and_labels(subset_train_files, threshold)
    pdb.set_trace()
    print("Step 2: Training Classifier")
    clf = train_classifier(features, labels)
    #save the classifier
    clf_path = os.path.join(output_dir, "rf_classifier.pkl")
    with open(clf_path, "wb") as f:
        import pickle
        pickle.dump(clf, f)
    print(f"Classifier saved to {clf_path}")
    pdb.set_trace()
    print("Step 3: Predicting and Saving Masks for Test Set (~5500 images)")
    predict_and_save(subset_test_files, clf, output_dir=output_dir)
    pdb.set_trace()
if __name__ == "__main__":
    main()
