#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Orthomosaic Job: ${JOB_NAME}, ID: ${JOB_ID}"
start_time=$(date +%s)

# Local directories inside the container
LOCAL_INPUT_DIR="/data/input"
LOCAL_PROCESSING_DIR="/data/processing/${JOB_NAME}_${JOB_ID}"
LOCAL_OUTPUT_DIR="/data/output"
LOCAL_LOG_DIR="/data/logs"
OM_FILENAME="${JOB_NAME}.tif"

mkdir -p "$LOCAL_INPUT_DIR" "$LOCAL_PROCESSING_DIR" "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR"

# --- 2. STAGE-IN: Download data from Cloud Storage ---
echo "Downloading input data from ${GCS_INPUT_PATH}"
gsutil -m cp -r "${GCS_INPUT_PATH}/*" "$LOCAL_INPUT_DIR/"

# --- 3. CORE LOGIC: Run the main processing script ---
# The script is now called with local container paths
./execution/single_job_om.sh "$JOB_NAME" "$LOCAL_INPUT_DIR" "$LOCAL_PROCESSING_DIR" "$(nproc)"

# Move and rename the final orthomosaic
if [ -f "$LOCAL_PROCESSING_DIR/code/odm_orthophoto/odm_orthophoto.tif" ]; then
    mv "$LOCAL_PROCESSING_DIR/code/odm_orthophoto/odm_orthophoto.tif" "$LOCAL_OUTPUT_DIR/$OM_FILENAME"
    echo "Orthomosaic created at $LOCAL_OUTPUT_DIR/$OM_FILENAME"
else
    echo "Error: Orthomosaic file not found in processing output!"
    exit 1
fi

# --- 4. STAGE-OUT: Upload results to Cloud Storage ---
echo "Uploading results to ${GCS_OUTPUT_DIR}"
gsutil -m cp "$LOCAL_OUTPUT_DIR/$OM_FILENAME" "${GCS_OUTPUT_DIR}/"

# --- 5. PERFORMANCE LOGGING ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Use 'conda run' to execute python in the correct environment
area_m2=$(conda run -n harvest python ./utils/get_om_area.py --ortho "$LOCAL_OUTPUT_DIR/$OM_FILENAME" --unit "m2")
output_size=$(du -sh "$LOCAL_OUTPUT_DIR/$OM_FILENAME" | awk '{print $1}')

# Write perf log locally, then upload
log_file="$LOCAL_LOG_DIR/execution_times_om.csv"
echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,area_m2,output_size" > "$log_file"
echo "$JOB_ID,$JOB_NAME,om,$num_cores,$MEMORY,$start_time,$end_time,$execution_time,$area_m2,$output_size" >> "$log_file"
gsutil -m cp "$log_file" "${GCS_PERF_LOG_PATH}/"

echo "Job finished successfully. Total time: $execution_time seconds"