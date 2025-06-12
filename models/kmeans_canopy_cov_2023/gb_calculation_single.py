import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.color import rgb2lab, deltaE_cie76
import os
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def visualize_classification(image_path, classification):
    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    color_map = {'brown': [0, 0, 0], 'green': [255, 255, 255], 'inbetween': [128, 128, 128], 'nodata': [194, 191, 199]}

    visualized = np.zeros((classification.shape[0], classification.shape[1], 3), dtype=np.uint8)
    edges = cv.Canny(cv.cvtColor(image, cv.COLOR_RGB2GRAY), 100, 200)

    for key, color in color_map.items():
        mask = classification == key
        visualized[mask] = color

    fig, axs = plt.subplots(1, 2, figsize=(20, 12))

    axs[1].imshow(visualized)
    axs[0].imshow(image)

    file_name = os.path.basename(image_path)

    plt.title(file_name)
    plt.show()
    plt.close()

def classify_pixels_with_kmeans(lab_image, kmeans):
    lab_image_ab = lab_image[:, :, 1:3]
    lab_image_flat = lab_image_ab.reshape(-1, 2)

    predicted_labels = kmeans.predict(lab_image_flat)

    classification = np.full(lab_image.shape[:2], '', dtype='<U10')

    predicted_labels = predicted_labels.reshape(lab_image.shape[:2])

    classification[predicted_labels == 1] = 'green'
    classification[predicted_labels == 0] = 'brown'

    return classification


def main():

    model_path = 'kmeans_model2.pkl'

    image_path = '/Users/hongchaeun/Library/CloudStorage/OneDrive-TheOhioStateUniversity/uas/plottiles/sony_optical_data_rev1/20230831_northwest_soy_410_sony.tif'

    image = io.imread(image_path)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    lab_image0 = rgb2lab(image)
    lab_image = rgb2lab(image)

    nodata_mask = np.all(lab_image0 == [0, 128, 128], axis=-1) | np.all(lab_image0 == [0, 0, 0], axis=-1)

    kmeans = joblib.load(model_path)
    classification = classify_pixels_with_kmeans(lab_image, kmeans)

    classification[nodata_mask] = 'nodata'

    h, w = classification.shape

    sections = []
    green_percentages = []

    for i in range(2): 
        for j in range(4):
            start_row = i * (h // 2)
            end_row = (i + 1) * (h // 2)
            start_col = j * (w // 4)
            end_col = (j + 1) * (w // 4)

            section = classification[start_row:end_row, start_col:end_col]
            sections.append(section)

            valid_pixels = np.sum(np.logical_or(section == 'green', section == 'brown'))
            green_pixels = np.sum(section == 'green')

            green_percentage = green_pixels / valid_pixels * 100 if valid_pixels > 0 else 0
            green_percentages.append(green_percentage)

    max_green = max(green_percentages)
    min_green = min(green_percentages)
    gap = max_green - min_green

    max_section = green_percentages.index(max_green)
    min_section = green_percentages.index(min_green)

    green_pixels = np.sum(classification == 'green')
    brown_pixels = np.sum(classification == 'brown')

    green_percentage = green_pixels / (green_pixels + brown_pixels) * 100
    file_name = os.path.basename(image_path)
    print(f"{file_name}: {green_percentage:.2f}")

    if classification is not None:
        visualize_classification(image_path, classification)

if __name__ == "__main__":
    main()
