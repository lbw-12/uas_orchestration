import os
import rasterio
import argparse
import shutil
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb
from collections import defaultdict
import re
from rasterio.transform import Affine

def get_patches_from_plots_rgb(plot_filename, output_path, patch_size = 224):
    
    with rasterio.open(plot_filename) as src:
        plot_data = src.read([1, 2, 3])  # (3, H, W)
        plot_meta = src.meta.copy()
        plot_transform = src.transform

        plot_data = plot_data.transpose(1, 2, 0)  # (H, W, 3)
        H, W = plot_data.shape[:2]
        print(f"Plot data shape: {plot_data.shape}")

        patches = []
        base_filename = os.path.basename(plot_filename).replace(".tif", "")

        patch_id = 0
        # Vertical positions: two patches with overlap
        vertical_starts = list(range(0, H - patch_size + 1, patch_size))
        if vertical_starts[-1] + patch_size < H:
            vertical_starts.append(H - patch_size)

        # Horizontal positions: slide without overlap, overlap at the end
        horizontal_starts = list(range(0, W - patch_size + 1, patch_size))
        if horizontal_starts[-1] + patch_size < W:
            horizontal_starts.append(W - patch_size)

        for y in vertical_starts:
            for x in horizontal_starts:
                patch = plot_data[y:y + patch_size, x:x + patch_size]
                patch_raster = np.transpose(patch, (2, 0, 1))

                # Update transform for the patch
                patch_transform = plot_transform * Affine.translation(x, y)

                # Update metadata
                patch_meta = plot_meta.copy()
                patch_meta.update({
                    "count": patch_raster.shape[0],
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform
                })

                patch_path = f"{output_path}/{base_filename}_{patch_id}.tif"

                with rasterio.open(patch_path, "w", **patch_meta) as dst:
                    dst.write(patch_raster)

                patch_id += 1
    print(f"Saved {patch_id} patches.")
    return patch_id

def main():
    parser = argparse.ArgumentParser(description="Generate patches from plot data")
    parser.add_argument("--plot_filename", type=str, required=True, help="Path to the plot raster file")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patches to be generated")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the patches")
    args = parser.parse_args()

    get_patches_from_plots_rgb(args.plot_filename, args.output_path, args.patch_size)

if __name__ == "__main__":
    main()
