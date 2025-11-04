#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Plot Tiles to Patches Job: ${JOB_NAME}, ID: ${JOB_ID}"
start_time=$(date +%s)

NUM_CORES=$(nproc)
echo "Number of cores: $NUM_CORES"

# --- 2. CONFIGURATION & DATA STAGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Platform: $PLATFORM. Configuring for cloud execution using rclone."
    JOB_IDENTIFIER="${JOB_ID}"
    JOB_MEMORY="${MEMORY}"
    LOCAL_INPUT_DIR="/data/input/plottiles"
    LOCAL_OUTPUT_DIR="/data/output/plot_patches"
    LOCAL_LOG_DIR="/data/logs"
    mkdir -p "$LOCAL_INPUT_DIR" "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR"
    
    EFFECTIVE_INPUT_PATH="$LOCAL_INPUT_DIR"
    EFFECTIVE_OUTPUT_PATH="$LOCAL_OUTPUT_DIR"
    
    echo "Downloading plot tiles from ${INPUT_PATH}"
    rclone copy "$INPUT_PATH" "$EFFECTIVE_INPUT_PATH"
    ;;
  slurm)
    echo "Platform: slurm. Configuring for local execution."
    JOB_IDENTIFIER="${SLURM_JOB_ID}"
    JOB_MEMORY="${SLURM_MEM_PER_NODE}"
    EFFECTIVE_INPUT_PATH="$INPUT_PATH"
    EFFECTIVE_OUTPUT_PATH="$OUTPUT_PATH_PLOT_PATCHES"
    mkdir -p "$EFFECTIVE_OUTPUT_PATH"
    ;;
  *)
    echo "Error: Unsupported PLATFORM value '${PLATFORM}'." >&2
    exit 1
    ;;
esac

echo "Job Identifier: ${JOB_IDENTIFIER}"
free -h

# --- 3. CORE LOGIC: Run the main Python script to convert plot tiles to patches ---
echo "Starting core plot tiles to patches processing"
echo "Input: ${EFFECTIVE_INPUT_PATH}, Output: ${EFFECTIVE_OUTPUT_PATH}"

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

conda run -n harvest python "$PYTHON_SCRIPT_PATH" \
    --input_dir "$EFFECTIVE_INPUT_PATH" \
    --output_dir "$EFFECTIVE_OUTPUT_PATH"

echo "Core plot tiles to patches processing finished."

# --- 4. PERFORMANCE LOGGING ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))
processing_step="plottile_to_patches_${SENSOR_TYPE}"

# Get the size of the output folder
output_size=$(du -sh "$EFFECTIVE_OUTPUT_PATH" | awk '{print $1}')

data_line="$JOB_IDENTIFIER,$JOB_TITLE,$processing_step,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$output_size"

# --- 5. STAGE-OUT & LOGGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Uploading results to ${OUTPUT_PATH_PLOT_PATCHES}"
    rclone copy "$EFFECTIVE_OUTPUT_PATH" "$OUTPUT_PATH_PLOT_PATCHES" --gcs-bucket-policy-only
    
    log_file="$LOCAL_LOG_DIR/perf_${JOB_IDENTIFIER}.csv"
    header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size"
    echo "$header" > "$log_file"
    echo "$data_line" >> "$log_file"
    echo "Uploading performance log to ${PERF_LOG_DIR}"
    rclone copy "$log_file" "$PERF_LOG_DIR" --gcs-bucket-policy-only
    ;;
  slurm)
    log_file="$PERF_LOG_DIR/execution_times_plottile_to_patches_${SENSOR_TYPE}.csv"
    if [ ! -f "$log_file" ]; then
      header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size"
      echo "$header" > "$log_file"
    fi
    echo "$data_line" >> "$log_file"
    echo "Performance log updated at $log_file"
    ;;
esac

echo "Job finished successfully. Total execution time: $execution_time seconds"

