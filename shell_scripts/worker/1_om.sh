#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. PLATFORM-AGNOSTIC SETUP ---
echo "Starting Universal Orthomosaic Job"
start_time=$(date +%s)

# Core variables expected from the orchestrator
# PLATFORM: "gcp", "aws", "azure", or "slurm"
# CORE_SCRIPT_NAME: e.g., "single_job_rgb.sh" or "single_job_ms.sh"
# JOB_NAME, INPUT_PATH, OUTPUT_DIR, PROCESSING_DIR, PERF_LOG_DIR, ODM_PATH

OM_FILENAME="${OM}_${SENSOR_TYPE}_${DATE}.tif"
echo "Orthomosaic filename will be: ${OM_FILENAME}"
NUM_CORES=$(nproc)

# --- 2. CONFIGURATION & DATA STAGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Platform: $PLATFORM. Configuring for cloud execution using rclone."
    JOB_IDENTIFIER="${JOB_ID}"
    JOB_MEMORY="${MEMORY}"
    LOCAL_INPUT_DIR="/data/input"
    LOCAL_OUTPUT_DIR="/data/output"
    LOCAL_LOG_DIR="/data/logs"
    mkdir -p "$LOCAL_INPUT_DIR" "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR"
    EFFECTIVE_INPUT_PATH="$LOCAL_INPUT_DIR"
    EFFECTIVE_OUTPUT_PATH="$LOCAL_OUTPUT_DIR"
    FINAL_OM_PATH="$EFFECTIVE_OUTPUT_PATH/$OM_FILENAME"
    echo "Downloading data from ${INPUT_PATH}"
    echo "Final orthomosaic will be saved to ${FINAL_OM_PATH}"
    rclone copy "$INPUT_PATH" "$EFFECTIVE_INPUT_PATH"
    ;;
  slurm)
    echo "Platform: slurm. Configuring for local execution."
    JOB_IDENTIFIER="${SLURM_JOB_ID}"
    JOB_MEMORY="${SLURM_MEM_PER_NODE}"
    EFFECTIVE_INPUT_PATH="$INPUT_PATH"
    EFFECTIVE_OUTPUT_PATH="$OUTPUT_DIR"
    FINAL_OM_PATH="$EFFECTIVE_OUTPUT_PATH/$OM_FILENAME"
    mkdir -p "$EFFECTIVE_OUTPUT_PATH" "$PROCESSING_DIR"
    LOCAL_ODM_PATH="$ODM_PATH"
    ;;
  *)
    echo "Error: Unsupported PLATFORM value '${PLATFORM}'." >&2
    exit 1
    ;;
esac

echo "Job Identifier: ${JOB_IDENTIFIER}"
free -h

# --- 3. CORE LOGIC (UNIFIED) ---
echo "Starting core orthomosaic processing with script: ${CORE_SCRIPT_NAME}"
echo "Input: ${EFFECTIVE_INPUT_PATH}, Processing: ${PROCESSING_DIR}"
${CORE_SCRIPT_NAME} "$JOB_NAME" "$EFFECTIVE_INPUT_PATH" "$PROCESSING_DIR" "$NUM_CORES" "$LOCAL_ODM_PATH"
echo "Core processing finished."

# --- 4. POST-PROCESSING (UNIFIED) ---
RAW_ORTHO_PATH="$PROCESSING_DIR/code/odm_orthophoto/odm_orthophoto.tif"
if [ -f "$RAW_ORTHO_PATH" ]; then
    mv "$RAW_ORTHO_PATH" "$FINAL_OM_PATH"
    echo "Orthomosaic moved to $FINAL_OM_PATH."
else
    echo "Error: Orthomosaic file not found at $RAW_ORTHO_PATH!"
    exit 1
fi

# --- 5. PERFORMANCE LOGGING (UNIFIED) ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# No 'case' statement is needed here if gdalinfo is in all environments
echo "Calculating area using GDAL..."
PIXEL_SIZE=$(gdalinfo "$FINAL_OM_PATH" | grep "Pixel Size" | sed -E 's/Pixel Size = \((.*),.*\)/\1/')
IMAGE_DIMS=$(gdalinfo "$FINAL_OM_PATH" | grep "Size is" | sed -E 's/Size is (.*), (.*)/\1*\2/')

# Use 'bc' for floating point math to calculate area in square meters
area_m2=$(echo "scale=2; $IMAGE_DIMS * $PIXEL_SIZE * $PIXEL_SIZE" | bc)

output_size=$(du -sh "$FINAL_OM_PATH" | awk '{print $1}')
data_line="$JOB_IDENTIFIER,$JOB_NAME,om,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$area_m2,$output_size"

# --- 6. STAGE-OUT & LOGGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Uploading results to ${OUTPUT_DIR}"
    rclone copy "$FINAL_OM_PATH" "$OUTPUT_DIR" --gcs-bucket-policy-only
    log_file="$LOCAL_LOG_DIR/perf_${JOB_IDENTIFIER}.csv"
    header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,area_m2,output_size"
    echo "$header" > "$log_file"
    echo "$data_line" >> "$log_file"
    echo "Uploading performance log to ${PERF_LOG_DIR}"
    rclone copy "$log_file" "$PERF_LOG_DIR" --gcs-bucket-policy-only
    ;;
  slurm)
    log_file="$PERF_LOG_DIR/execution_times_om.csv"
    if [ ! -f "$log_file" ]; then
      header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,area_m2,output_size"
      echo "$header" > "$log_file"
    fi
    echo "$data_line" >> "$log_file"
    echo "Performance log updated at $log_file"
    ;;
esac

echo "Job finished successfully. Total execution time: $execution_time seconds"