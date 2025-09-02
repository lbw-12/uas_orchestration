Bash
#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Orthomosaic Alignment Job: ${JOB_NAME}, ID: ${JOB_ID}"
start_time=$(date +%s)

# Local directories inside the container
LOCAL_OM_DIR="/data/input/om"
LOCAL_SHAPEFILE_DIR="/data/input/shapefiles"
LOCAL_ALIGNED_DIR="/data/output/aligned"
LOCAL_LOG_DIR="/data/logs"
ALIGNED_OM_FILENAME="${JOB_NAME}_aligned.tif"

mkdir -p "$LOCAL_OM_DIR" "$LOCAL_SHAPEFILE_DIR" "$LOCAL_ALIGNED_DIR" "$LOCAL_LOG_DIR"

# --- 2. STAGE-IN: Download data from Cloud Storage ---
echo "Downloading orthomosaics from ${GCS_OM_FOLDER}"
gsutil -m cp -r "${GCS_OM_FOLDER}/*" "$LOCAL_OM_DIR/"

echo "Downloading shapefiles from ${GCS_SHAPEFILE_PATH}"
gsutil -m cp -r "${GCS_SHAPEFILE_PATH}/*" "$LOCAL_SHAPEFILE_DIR/"

# --- 3. CORE LOGIC: Run the main Python alignment script ---
echo "Running alignment script..."
conda run -n harvest python om_alignment_old.py \
    --search "${SEARCH_PATTERN}" \
    --pattern "${OM_PATTERN}" \
    --shapefile_path "$LOCAL_SHAPEFILE_DIR" \
    --om_folder "$LOCAL_OM_DIR" \
    --om_aligned_folder "$LOCAL_ALIGNED_DIR"

# --- 4. STAGE-OUT: Upload results to Cloud Storage ---
echo "Uploading aligned orthomosaic to ${GCS_ALIGNED_FOLDER}"
gsutil -m cp "$LOCAL_ALIGNED_DIR/$ALIGNED_OM_FILENAME" "${GCS_ALIGNED_FOLDER}/"

# --- 5. PERFORMANCE LOGGING ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Calculate alignment points and log performance
alignment_points=$(conda run -n harvest python ./utils/get_alignment_points.py "${LOCAL_SHAPEFILE_DIR}/${OM_PATTERN}_pts/${OM_PATTERN}_pts.shp")
output_size=$(du -sh "$LOCAL_ALIGNED_DIR/$ALIGNED_OM_FILENAME" | awk '{print $1}')

log_file="$LOCAL_LOG_DIR/execution_times_omalign.csv"
echo "job_id,job_name,..." > "$log_file" # Add headers
echo "$JOB_ID,$JOB_NAME,..." >> "$log_file" # Add data
gsutil -m cp "$log_file" "${GCS_PERF_LOG_PATH}/"

echo "Job finished successfully. Total time: $execution_time seconds"