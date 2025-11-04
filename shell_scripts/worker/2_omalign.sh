#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Orthomosaic Alignment Job: ${JOB_NAME}, ID: ${JOB_ID}"
start_time=$(date +%s)

ALIGN_FILENAME="${OM}_${SENSOR_TYPE}_${DATE}_aligned.tif"
NUM_CORES=$(nproc)

# --- 2. CONFIGURATION & DATA STAGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Platform: $PLATFORM. Configuring for cloud execution using rclone."
    JOB_IDENTIFIER="${JOB_ID}"
    JOB_MEMORY="${MEMORY}"
    LOCAL_OM_DIR="/data/input/om"
    LOCAL_SHAPEFILE_DIR="/data/input/shapefiles"
    LOCAL_ALIGNED_DIR="/data/output/aligned"
    LOCAL_LOG_DIR="/data/logs"
    mkdir -p "$LOCAL_OM_DIR" "$LOCAL_SHAPEFILE_DIR" "$LOCAL_ALIGNED_DIR" "$LOCAL_LOG_DIR"
    EFFECTIVE_OM_PATH="$LOCAL_OM_DIR"
    EFFECTIVE_SHAPEFILE_PATH="$LOCAL_SHAPEFILE_DIR"
    EFFECTIVE_OUTPUT_PATH="$LOCAL_ALIGNED_DIR"
    FINAL_ALIGNED_PATH="$EFFECTIVE_OUTPUT_PATH/$ALIGN_FILENAME"
    
    echo "Downloading orthomosaics from ${OM_INPUT_PATH}"
    rclone copy "$OM_INPUT_PATH" "$EFFECTIVE_OM_PATH"
    echo "Downloading shapefiles from ${SHAPEFILE_INPUT_PATH}"
    rclone copy "$SHAPEFILE_INPUT_PATH" "$EFFECTIVE_SHAPEFILE_PATH"
    ;;
  slurm)
    echo "Platform: slurm. Configuring for local execution."
    JOB_IDENTIFIER="${SLURM_JOB_ID}"
    JOB_MEMORY="${SLURM_MEM_PER_NODE}"
    EFFECTIVE_OM_PATH="$OM_INPUT_PATH"
    EFFECTIVE_SHAPEFILE_PATH="$SHAPEFILE_INPUT_PATH"
    EFFECTIVE_OUTPUT_PATH="$OUTPUT_DIR"
    FINAL_ALIGNED_PATH="$EFFECTIVE_OUTPUT_PATH/$ALIGN_FILENAME"
    mkdir -p "$EFFECTIVE_OUTPUT_PATH"
    ;;
  *)
    echo "Error: Unsupported PLATFORM value '${PLATFORM}'." >&2
    exit 1
    ;;
esac

echo "Job Identifier: ${JOB_IDENTIFIER}"
free -h

# --- 3. CORE LOGIC: Run the main Python alignment script ---
echo "Starting core alignment processing"
echo "OM Input: ${EFFECTIVE_OM_PATH}, Shapefile Input: ${EFFECTIVE_SHAPEFILE_PATH}, Output: ${EFFECTIVE_OUTPUT_PATH}"

# Platform-specific Python script path
case "$PLATFORM" in
  gcp|aws|azure)
    PYTHON_SCRIPT="/app/execution/om_alignment_old.py"
    ;;
  slurm)
    PYTHON_SCRIPT="./om_alignment_old.py"  # Relative path for SLURM
    ;;
esac

echo "Using Python script: $PYTHON_SCRIPT"

conda run -n harvest python "$PYTHON_SCRIPT" \
    --search "${SEARCH_PATTERN}" \
    --pattern "${OM_PATTERN}" \
    --shapefile_path "$EFFECTIVE_SHAPEFILE_PATH" \
    --om_folder "$EFFECTIVE_OM_PATH" \
    --om_aligned_folder "$EFFECTIVE_OUTPUT_PATH"

echo "Core alignment processing finished."

# --- 4. POST-PROCESSING (UNIFIED) ---
RAW_ALIGNED_PATH="$EFFECTIVE_OUTPUT_PATH/$ALIGN_FILENAME"
if [ -f "$RAW_ALIGNED_PATH" ]; then
    echo "Aligned orthomosaic found at $RAW_ALIGNED_PATH."
    FINAL_ALIGNED_PATH="$RAW_ALIGNED_PATH"
else
    echo "Error: Aligned orthomosaic file not found at $RAW_ALIGNED_PATH!"
    exit 1
fi

# --- 5. PERFORMANCE LOGGING (UNIFIED) ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Try to get alignment points, but don't fail if script doesn't exist
if [ -f "./utils/get_alignment_points.py" ]; then
    alignment_points=$(conda run -n harvest python ./utils/get_alignment_points.py "${EFFECTIVE_SHAPEFILE_PATH}/${OM_PATTERN}_pts/${OM_PATTERN}_pts.shp" 2>/dev/null || echo "0")
else
    echo "Alignment points script not found, skipping count"
    alignment_points="0"
fi

output_size=$(du -sh "$FINAL_ALIGNED_PATH" | awk '{print $1}')
data_line="$JOB_IDENTIFIER,$JOB_NAME,omalign,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$alignment_points,$output_size"

# --- 6. STAGE-OUT & LOGGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Uploading results to ${OUTPUT_DIR}"
    rclone copy "$FINAL_ALIGNED_PATH" "$OUTPUT_DIR" --gcs-bucket-policy-only
    log_file="$LOCAL_LOG_DIR/perf_${JOB_IDENTIFIER}.csv"
    header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,alignment_points,output_size"
    echo "$header" > "$log_file"
    echo "$data_line" >> "$log_file"
    echo "Uploading performance log to ${PERF_LOG_DIR}"
    rclone copy "$log_file" "$PERF_LOG_DIR" --gcs-bucket-policy-only
    ;;
  slurm)
    log_file="$PERF_LOG_DIR/execution_times_omalign.csv"
    if [ ! -f "$log_file" ]; then
      header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,alignment_points,output_size"
      echo "$header" > "$log_file"
    fi
    echo "$data_line" >> "$log_file"
    echo "Performance log updated at $log_file"
    ;;
esac

echo "Job finished successfully. Total execution time: $execution_time seconds"