#!/bin/bash

# Function to run OpenDroneMap with the specified options
job=$1
img_dir=$2
processing_subdir=$3
num_cores=$4

echo "job: $job"
echo "img_dir: $img_dir"
echo "processing_subdir: $processing_subdir"
echo "num_cores: $num_cores"

# Path to the ODM image
# odm_image_path=/fs/ess/PAS2699/script_preprocessing/odm_gpu.sif
odm_image_path=/fs/ess/PAS2699/odm_latest.sif
log_dir="./log/"
mkdir -p "$log_dir"

code_dir="$processing_subdir/code/images"

# Run OpenDroneMap with the specified options and measure execution time
start_time=$(date +%s)

echo "Starting OpenDroneMap processing..."
echo "Input directory: $img_dir"
echo "Output directory: $processing_subdir"

# Apptainer has issue to bind the directory with space in the name as workaround we will copy the images to the code directory
echo "Copying images to ODM directory"
mkdir -p "$code_dir"

# Move only top-level files (ignore subdirectories)
find "$img_dir" -maxdepth 1 -type f -exec cp {} "$code_dir" \;

echo "Images copied to code directory"
echo "Starting ODM"
time apptainer run \
    --writable-tmpfs "$odm_image_path" \
    --mesh-octree-depth 9 --dtm --force-gps --use-exif --debug\
    --smrf-threshold 0.5 --smrf-window 18 --dsm --pc-csv --pc-las --orthophoto-kmz \
    --matcher-type flann --feature-quality high --max-concurrency 8 \
    --build-overviews --min-num-features 8000 --skip-3dmodel --dem-gapfill-steps 0 \
    --orthophoto-resolution 0.7 --ignore-gsd --mesh-size 50000 --dem-resolution 1.5 --radiometric-calibration none \
    --project-path "$processing_subdir"

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Processing completed."
echo "Total execution time: $execution_time seconds"

echo "Removing Duplicated images from code directory"
rm -r "$code_dir"