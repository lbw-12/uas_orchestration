#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Spectral Reflectance Inference Job: ${JOB_NAME}, ID: ${JOB_ID}"
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
    LOCAL_MODEL_DIR="/data/models"
    LOCAL_OUTPUT_DIR="/data/output/inference"
    LOCAL_LOG_DIR="/data/logs"
    mkdir -p "$LOCAL_INPUT_DIR" "$LOCAL_MODEL_DIR" "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR"
    
    EFFECTIVE_INPUT_DIR="$LOCAL_INPUT_DIR"
    EFFECTIVE_OUTPUT_DIR="$LOCAL_OUTPUT_DIR"
    EFFECTIVE_OUTPUT_JSON="$LOCAL_OUTPUT_DIR/$(basename $OUTPUT_PATH_SR_JSON)"

    echo "Downloading plot tiles from ${INPUT_DIR}"
    rclone copy "$INPUT_DIR" "$EFFECTIVE_INPUT_DIR"

    # Check if model is already in container or needs to be downloaded
    if [[ "$MODEL_PATH" == /app/* ]]; then
      echo "Model is already in container at ${MODEL_PATH}"
      EFFECTIVE_MODEL_PATH="$MODEL_PATH"
    else
      echo "Downloading model from ${MODEL_PATH}"
      EFFECTIVE_MODEL_PATH="$LOCAL_MODEL_DIR/$(basename $MODEL_PATH)"
      rclone copy "$MODEL_PATH" "$EFFECTIVE_MODEL_PATH"
    fi
    ;;
  slurm)
    echo "Platform: slurm. Configuring for local execution."
    JOB_IDENTIFIER="${SLURM_JOB_ID}"
    JOB_MEMORY="${SLURM_MEM_PER_NODE}"
    EFFECTIVE_INPUT_DIR="$INPUT_DIR"
    EFFECTIVE_OUTPUT_DIR="$OUTPUT_PATH_SR"
    EFFECTIVE_OUTPUT_JSON="$OUTPUT_PATH_SR_JSON"
    EFFECTIVE_MODEL_PATH="$MODEL_PATH"
    mkdir -p "$EFFECTIVE_OUTPUT_DIR"
    mkdir -p "$(dirname $EFFECTIVE_OUTPUT_JSON)"
    ;;
  *)
    echo "Error: Unsupported PLATFORM value '${PLATFORM}'." >&2
    exit 1
    ;;
esac

echo "Job Identifier: ${JOB_IDENTIFIER}"
free -h

# --- 3. CORE LOGIC: Run the main Python script for spectral reflectance inference ---
echo "Starting spectral reflectance inference"
echo "Input: ${EFFECTIVE_INPUT_DIR}, Output Dir: ${EFFECTIVE_OUTPUT_DIR}, Output JSON: ${EFFECTIVE_OUTPUT_JSON}, Model: ${EFFECTIVE_MODEL_PATH}"

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
    --input_dir "$EFFECTIVE_INPUT_DIR" \
    --output_dir "$EFFECTIVE_OUTPUT_DIR" \
    --output_json "$EFFECTIVE_OUTPUT_JSON" \
    --model_path "$EFFECTIVE_MODEL_PATH" \
    --field "$OM" \
    --plotimage_source "$PLOTIMAGE_SOURCE" \
    --date "$DATE"

echo "Spectral reflectance inference finished."

# --- 4. PERFORMANCE LOGGING ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))
processing_step="spectral_reflectance_inference"

# Get the size of the output folder
output_size=$(du -sh "$EFFECTIVE_OUTPUT_DIR" | awk '{print $1}')

data_line="$JOB_IDENTIFIER,$JOB_TITLE,$processing_step,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$output_size"

# --- 5. STAGE-OUT & LOGGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Uploading results to ${OUTPUT_PATH_SR}"
    rclone copy "$EFFECTIVE_OUTPUT_DIR" "$OUTPUT_PATH_SR" --gcs-bucket-policy-only
    
    echo "Uploading JSON to ${OUTPUT_PATH_SR_JSON}"
    rclone copy "$EFFECTIVE_OUTPUT_JSON" "$(dirname $OUTPUT_PATH_SR_JSON)" --gcs-bucket-policy-only
    
    log_file="$LOCAL_LOG_DIR/perf_${JOB_IDENTIFIER}.csv"
    header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size"
    echo "$header" > "$log_file"
    echo "$data_line" >> "$log_file"
    echo "Uploading performance log to ${PERF_LOG_DIR}"
    rclone copy "$log_file" "$PERF_LOG_DIR" --gcs-bucket-policy-only
    ;;
  slurm)
    log_file="$PERF_LOG_DIR/execution_times_spectral_reflectance.csv"
    if [ ! -f "$log_file" ]; then
      header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size"
      echo "$header" > "$log_file"
    fi
    echo "$data_line" >> "$log_file"
    echo "Performance log updated at $log_file"
    ;;
esac

echo "Job finished successfully. Total execution time: $execution_time seconds"

