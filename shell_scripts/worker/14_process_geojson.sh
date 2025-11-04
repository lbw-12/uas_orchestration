#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Process GeoJSON Job: ${JOB_NAME}, ID: ${JOB_ID}"
start_time=$(date +%s)

NUM_CORES=$(nproc)
echo "Number of cores: $NUM_CORES"

# --- 2. CONFIGURATION & DATA STAGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Platform: $PLATFORM. Configuring for cloud execution using rclone."
    JOB_IDENTIFIER="${JOB_ID}"
    JOB_MEMORY="${MEMORY}"
    LOCAL_INPUT_DIR="/data/input/inference"
    LOCAL_OUTPUT_DIR="/data/output/geojson"
    LOCAL_LOG_DIR="/data/logs"
    LOCAL_BASE_DIR="/data/base"
    mkdir -p "$LOCAL_INPUT_DIR" "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR" "$LOCAL_BASE_DIR"

    EFFECTIVE_INFERENCE_FOLDER="$LOCAL_INPUT_DIR"
    EFFECTIVE_OUTPUT_JSON="$LOCAL_OUTPUT_DIR/$(basename $OUTPUT_FOLDER_GEOJSON)"
    EFFECTIVE_BASE_FOLDER="$LOCAL_BASE_DIR"

    echo "Downloading inference results from ${JSON_INFERENCE_DIR}"
    rclone copy "$JSON_INFERENCE_DIR" "$EFFECTIVE_INFERENCE_FOLDER"

    # Download shapefiles needed by process_geojson.py
    echo "Downloading shapefiles from ${BASE_FOLDER}shapefiles/"
    rclone copy "${BASE_FOLDER}shapefiles/" "$LOCAL_BASE_DIR/shapefiles/" --include "*.shp" --include "*.shx" --include "*.dbf" --include "*.prj" --include "*.cpg"
    ;;
  slurm)
    echo "Platform: slurm. Configuring for local execution."
    JOB_IDENTIFIER="${SLURM_JOB_ID}"
    JOB_MEMORY="${SLURM_MEM_PER_NODE}"
    EFFECTIVE_INFERENCE_FOLDER="$JSON_INFERENCE_DIR"
    EFFECTIVE_OUTPUT_JSON="$OUTPUT_FOLDER_GEOJSON"
    EFFECTIVE_BASE_FOLDER="$BASE_FOLDER"
    mkdir -p "$(dirname $EFFECTIVE_OUTPUT_JSON)"
    ;;
  *)
    echo "Error: Unsupported PLATFORM value '${PLATFORM}'." >&2
    exit 1
    ;;
esac

echo "Job Identifier: ${JOB_IDENTIFIER}"
free -h

# --- 3. CORE LOGIC: Run the main Python script to generate GeoJSON ---
echo "Starting GeoJSON generation"
echo "Inference folder: ${EFFECTIVE_INFERENCE_FOLDER}, Output: ${EFFECTIVE_OUTPUT_JSON}"

# Platform-specific Python script path
case "$PLATFORM" in
  gcp|aws|azure)
    PYTHON_SCRIPT_PATH="/app/execution/${PYTHON_SCRIPT}"
    ;;
  slurm)
    PYTHON_SCRIPT_PATH="./${PYTHON_SCRIPT}"
    ;;
esac

echo "Using Python script: $PYTHON_SCRIPT_PATH"

# Pass config values based on platform
case "$PLATFORM" in
  gcp|aws|azure)
    # For GCP, pass config values as arguments with local paths
    conda run -n harvest python "$PYTHON_SCRIPT_PATH" \
        --inference_folder "$EFFECTIVE_INFERENCE_FOLDER" \
        --location "$OM" \
        --date "$DATE" \
        --output_json "$EFFECTIVE_OUTPUT_JSON" \
        --base_folder "$EFFECTIVE_BASE_FOLDER" \
        --maptiles_folder "$MAPTILES_FOLDER" \
        --plot_shapefiles_json "$PLOT_SHAPEFILES_JSON"
    ;;
  slurm)
    # For SLURM, pass config values as arguments
    conda run -n harvest python "$PYTHON_SCRIPT_PATH" \
        --inference_folder "$EFFECTIVE_INFERENCE_FOLDER" \
        --location "$OM" \
        --date "$DATE" \
        --output_json "$EFFECTIVE_OUTPUT_JSON" \
        --base_folder "$EFFECTIVE_BASE_FOLDER" \
        --maptiles_folder "$MAPTILES_FOLDER" \
        --plot_shapefiles_json "$PLOT_SHAPEFILES_JSON"
    ;;
esac

echo "GeoJSON generation finished."

# --- 4. PERFORMANCE LOGGING ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))
processing_step="generating_geojson"

# Get the size of the output folder
output_size=$(du -sh "$EFFECTIVE_INFERENCE_FOLDER" | awk '{print $1}')

data_line="$JOB_IDENTIFIER,$JOB_TITLE,$processing_step,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$output_size"

# --- 5. STAGE-OUT & LOGGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Uploading GeoJSON to ${OUTPUT_FOLDER_GEOJSON}"
    rclone copy "$EFFECTIVE_OUTPUT_JSON" "$(dirname $OUTPUT_FOLDER_GEOJSON)" --gcs-bucket-policy-only
    
    log_file="$LOCAL_LOG_DIR/perf_${JOB_IDENTIFIER}.csv"
    header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size"
    echo "$header" > "$log_file"
    echo "$data_line" >> "$log_file"
    echo "Uploading performance log to ${PERF_LOG_DIR}"
    rclone copy "$log_file" "$PERF_LOG_DIR" --gcs-bucket-policy-only
    ;;
  slurm)
    log_file="$PERF_LOG_DIR/execution_times_geojson.csv"
    if [ ! -f "$log_file" ]; then
      header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size"
      echo "$header" > "$log_file"
    fi
    echo "$data_line" >> "$log_file"
    echo "Performance log updated at $log_file"
    ;;
esac

echo "Job finished successfully. Total execution time: $execution_time seconds"

