import os
import numpy as np
from skimage import io
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import joblib
from colorspacious import cspace_convert

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

    return classification, image

def calculate_green_percentage(classification):
    green_pixels = np.sum(classification == 'green')
    valid_pixels = np.sum(np.logical_or(classification == 'green', classification == 'brown'))
    return (green_pixels / (valid_pixels + 1e-6)) * 100

def visualize_classification(image, classification, out_path, canopy_percentage=None):
    color_map = {
        'brown': [0, 0, 0],
        'green': [255, 255, 255],
        'nodata': [194, 191, 199]
    }

    # Map classification labels to colors
    visualized = np.zeros((classification.shape[0], classification.shape[1], 3), dtype=np.uint8)
    for key, val in color_map.items():
        visualized[classification == key] = val

    # Create figure with absolute control
    fig = plt.figure(figsize=(12, 6), dpi=300)

    # Position top image (Original)
    ax1 = fig.add_axes([0.0, 0.52, 1.0, 0.45])  # [left, bottom, width, height]
    ax1.imshow(image)
    ax1.set_title("Original RGB Image", fontsize=11, pad=1)
    ax1.axis('off')

    # Position bottom image (Classified)
    ax2 = fig.add_axes([0.0, 0.07, 1.0, 0.45])
    ax2.imshow(visualized)
    ax2.set_title("Classified Image", fontsize=11, pad=1)
    ax2.axis('off')

    # Text tightly below second image
    if canopy_percentage is not None:
        fig.text(0.5, 0.005, f"Canopy Cover: {canopy_percentage:.2f}%", ha='center', fontsize=9)

    # Save with zero padding
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # === MODIFY THESE PATHS ===
    image_path = "/fs/ess/PAS2699/nitrogen/data/uas/2024/plottiles/plot_tiles_rgb_om/northwest_bftb_om_corn_rgb_218_20240805.tif"
    model_path = "minibatch_kmeans_model.pkl"
    output_path = "visualized_output.png"

    # === PROCESS ===
    kmeans = load_model(model_path)
    classification, image = classify_image(image_path, kmeans)
    canopy_percentage = calculate_green_percentage(classification)
    visualize_classification(image, classification, output_path, canopy_percentage)
    print(f"Saved visualization to: {output_path}")
    print(f"Canopy coverage: {canopy_percentage:.2f}%")
