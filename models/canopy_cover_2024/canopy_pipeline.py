import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import io, color
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import joblib
import argparse
import json
import cv2 as cv
from colorspacious import cspace_convert
import pdb

def load_images_in_batches(image_dir, batch_size=50000, file_ext=".tif"):
    batch = []
    count = 0
    for filename in tqdm(os.listdir(image_dir), desc="Loading images"):
        if filename.endswith(file_ext) and not filename.startswith("._"):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = np.array(img)
            pixels = img[:, :, :3].reshape(-1, 3)
            batch.extend(pixels)
            if len(batch) >= batch_size:
                yield np.array(batch[:batch_size])
                batch = batch[batch_size:]
            count += 1
            if count > 1000:
                break
    if batch:
        yield np.array(batch)

def fit_kmeans_on_batches(image_dir, n_clusters=2, batch_size=50000, model_path="minibatch_kmeans_model.pkl"):
    
    print("Fitting MiniBatchKMeans on LAB color space...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size)

    for batch in load_images_in_batches(image_dir, batch_size):
        lab = cspace_convert(batch / 255.0, "sRGB1", "CIELab")
        lab_ab = lab[:, 1:3]  # Only use a* and b*
        kmeans.partial_fit(lab_ab)
        
    joblib.dump(kmeans, model_path)
    print(f"Model saved to {model_path}")
    return kmeans

def classify_image(image_path, kmeans):
    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    lab = rgb2lab(image)
    nodata_mask = np.all(lab == [0, 128, 128], axis=-1) | np.all(lab == [0, 0, 0], axis=-1)

    lab_ab = lab[:, :, 1:3].reshape(-1, 2)
    labels = kmeans.predict(lab_ab).reshape(lab.shape[:2])

    # Determine cluster labels from centroids
    centroids = kmeans.cluster_centers_
    green_cluster, brown_cluster = sorted(range(2), key=lambda i: centroids[i][0])

    classification = np.full(lab.shape[:2], '', dtype='<U10')
    classification[labels == green_cluster] = 'green'
    classification[labels == brown_cluster] = 'brown'
    classification[nodata_mask] = 'nodata'
    return classification, image

def visualize_classification(image, classification, out_path):
    color_map = {
        'brown': [0, 0, 0],
        'green': [255, 255, 255],
        'nodata': [194, 191, 199]
    }

    visualized = np.zeros((classification.shape[0], classification.shape[1], 3), dtype=np.uint8)
    for key, color_val in color_map.items():
        mask = classification == key
        visualized[mask] = color_val

    fig, axs = plt.subplots(1, 2, figsize=(20, 12))
    axs[0].imshow(image)
    axs[1].imshow(visualized)
    axs[0].set_title("Original")
    axs[1].set_title("Classification")

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_green_percentage(classification, grid=(2, 4)):
    h, w = classification.shape
    section_stats = []

    for i in range(grid[0]):
        for j in range(grid[1]):
            sr, er = i * (h // grid[0]), (i + 1) * (h // grid[0])
            sc, ec = j * (w // grid[1]), (j + 1) * (w // grid[1])
            section = classification[sr:er, sc:ec]

            green_pixels = np.sum(section == 'green')
            valid_pixels = np.sum(np.logical_or(section == 'green', section == 'brown'))
            green_percent = green_pixels / valid_pixels * 100 if valid_pixels else 0

            section_stats.append(green_percent)

    return {
        "overall_green_percentage": float(np.sum(classification == 'green')) / (
            np.sum(np.logical_or(classification == 'green', classification == 'brown')) + 1e-6) * 100,
        "section_green_percentages": section_stats,
        "max_section": int(np.argmax(section_stats)),
        "min_section": int(np.argmin(section_stats)),
        "green_pixel_count": int(np.sum(classification == 'green')),
        "brown_pixel_count": int(np.sum(classification == 'brown'))
    }

def main(args):
    
    if args.train:
        kmeans = fit_kmeans_on_batches(args.input_dir, args.n_clusters, args.batch_size, args.model_path)
    else:
        kmeans = joblib.load(args.model_path)

    output_json = {}
    for fname in tqdm(os.listdir(args.input_dir), desc="Processing images"):
        if fname.endswith(args.file_ext) and not fname.startswith("._"):
            image_path = os.path.join(args.input_dir, fname)
            classification, image = classify_image(image_path, kmeans)
            stats = calculate_green_percentage(classification)
            output_json[fname] = stats

            if args.visualize:
                vis_path = os.path.join(args.output_dir, f"visualized_{fname}.png")
                visualize_classification(image, classification, vis_path)
            
    json_path = os.path.join(args.output_dir, "canopy_cover_summary.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved summary to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canopy Cover Clustering Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .tif images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_path", type=str, default="minibatch_kmeans_model.pkl", help="Path to save/load model")
    parser.add_argument("--train", action="store_true", help="Train KMeans model")
    parser.add_argument("--n_clusters", type=int, default=2, help="Number of clusters (default 2)")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size for MiniBatchKMeans")
    parser.add_argument("--file_ext", type=str, default=".tif", help="Image file extension to process")
    parser.add_argument("--visualize", action="store_true", help="Save visualized classification images")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
