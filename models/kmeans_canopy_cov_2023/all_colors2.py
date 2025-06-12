import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import os
from colorspacious import cspace_convert
import joblib

palette_dir = "/Users/hongchaeun/Desktop/growth_stage/color_palette/"
palettes = []
print("Loading images...")
for filename in os.listdir(palette_dir):
    if filename.endswith(".tif") and not filename.startswith("._"):
        img_path = os.path.join(palette_dir, filename)
        img = Image.open(img_path)
        img = np.array(img)
        palettes.append(img.reshape(-1, 3))
print(f"Loaded {len(palettes)} images.")

all_colors = np.vstack(palettes)
print(f"Total colors loaded: {all_colors.shape[0]}")

print("Converting RGB colors to LAB...")
lab_colors = cspace_convert(all_colors / 255.0, "sRGB1", "CIELab")

print("Applying PCA to LAB colors...")
pca = PCA(n_components=2)
lab_ab_pca = pca.fit_transform(lab_colors[:, 1:3])

print("Performing KMeans clustering...")
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(lab_ab_pca)
labels = kmeans.labels_

centroids = kmeans.cluster_centers_

clusters = sorted(range(2), key=lambda i: centroids[i][0])

cluster_names = ['green', 'brown']
green_cluster, brown_cluster = clusters

print(f"Green cluster index: {green_cluster}")
print(f"Brown cluster index: {brown_cluster}")

green_colors = all_colors[labels == green_cluster]
brown_colors = all_colors[labels == brown_cluster]

print(f"Green colors: {green_colors.shape[0]}")
print(f"Brown colors: {brown_colors.shape[0]}")

unique_green_lab_colors = np.unique(lab_colors[labels == green_cluster], axis=0)
unique_brown_lab_colors = np.unique(lab_colors[labels == brown_cluster], axis=0)

print("Saving KMeans model for later use...")
joblib.dump(kmeans, "kmeans_model2.pkl")

# plt.figure(figsize=(10, 8))
# plt.scatter(lab_ab_pca[labels == green_cluster, 0], lab_ab_pca[labels == green_cluster, 1],
#             color='green', label='Green Cluster', alpha=0.5, s=5)
# plt.scatter(lab_ab_pca[labels == brown_cluster, 0], lab_ab_pca[labels == brown_cluster, 1],
#             color='brown', label='Brown Cluster', alpha=0.5, s=5)
# plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=200, label='Centroids')
# plt.title('2D Visualization of Clusters in PCA Space')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()
# plt.show()

print("Process complete.")
