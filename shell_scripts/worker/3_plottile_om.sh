#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Plot Tile Extraction Job: ${JOB_NAME}, ID: ${JOB_ID}"
start_time=$(date +%s)

NUM_CORES=$(nproc)
echo "Number of cores: $NUM_CORES"

# --- 2. CONFIGURATION & DATA STAGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Platform: $PLATFORM. Configuring for cloud execution using rclone."
    JOB_IDENTIFIER="${JOB_ID}"
    JOB_MEMORY="${MEMORY}"
    LOCAL_CSV_DIR="/data/input/csv"
    LOCAL_OM_DIR="/data/input/aligned"
    LOCAL_SHAPEFILE_DIR="/data/input/shapefiles"
    LOCAL_OUTPUT_DIR="/data/output/plottiles"
    LOCAL_LOG_DIR="/data/logs"
    mkdir -p "$LOCAL_CSV_DIR" "$LOCAL_OM_DIR" "$LOCAL_SHAPEFILE_DIR" "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR"
    
    EFFECTIVE_CSV_PATH="$LOCAL_CSV_DIR"
    EFFECTIVE_OM_PATH="$LOCAL_OM_DIR"
    EFFECTIVE_SHAPEFILE_PATH="$LOCAL_SHAPEFILE_DIR"
    EFFECTIVE_OUTPUT_PATH="$LOCAL_OUTPUT_DIR"
    
    echo "Downloading CSV files from ${CSV_FOLDER_PATH}"
    rclone copy "$CSV_FOLDER_PATH" "$EFFECTIVE_CSV_PATH" --include "*.csv"
    echo "Downloading aligned orthomosaic from ${OM_ALIGNED_FOLDER}"
    rclone copy "${OM_ALIGNED_FOLDER}${JOB_TITLE}_aligned.tif" "$EFFECTIVE_OM_PATH"
    echo "Downloading shapefiles from ${SHAPEFILE_PATH1%/*}"  # Get directory from path
    rclone copy "${SHAPEFILE_PATH1%/*}" "$EFFECTIVE_SHAPEFILE_PATH"
    # Also download from SHAPEFILE_PATH2 directory if it's different and exists
    if [ -n "$SHAPEFILE_PATH2" ] && [ "${SHAPEFILE_PATH2%/*}" != "${SHAPEFILE_PATH1%/*}" ]; then
        echo "Downloading additional shapefiles from ${SHAPEFILE_PATH2%/*}"
        rclone copy "${SHAPEFILE_PATH2%/*}" "$EFFECTIVE_SHAPEFILE_PATH" --include "*.shp" --include "*.shx" --include "*.dbf" --include "*.prj" --include "*.cpg"
    fi
    ;;
  slurm)
    echo "Platform: slurm. Configuring for local execution."
    JOB_IDENTIFIER="${SLURM_JOB_ID}"
    JOB_MEMORY="${SLURM_MEM_PER_NODE}"
    EFFECTIVE_CSV_PATH="$CSV_FOLDER_PATH"
    EFFECTIVE_OM_PATH="$OM_ALIGNED_FOLDER"
    EFFECTIVE_SHAPEFILE_PATH="${SHAPEFILE_PATH1%/*}"
    EFFECTIVE_OUTPUT_PATH="$OUTPUT_PATH_PLOTTILES"
    mkdir -p "$EFFECTIVE_OUTPUT_PATH"
    ;;
  *)
    echo "Error: Unsupported PLATFORM value '${PLATFORM}'." >&2
    exit 1
    ;;
esac

echo "Job Identifier: ${JOB_IDENTIFIER}"
free -h

# --- 3. CORE LOGIC: Run the main Python plot tile extraction script ---
echo "Starting core plot tile extraction processing"
echo "CSV Input: ${EFFECTIVE_CSV_PATH}, OM Input: ${EFFECTIVE_OM_PATH}, Shapefile Input: ${EFFECTIVE_SHAPEFILE_PATH}, Output: ${EFFECTIVE_OUTPUT_PATH}"

# Set GDAL environment variable for robust file access
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Platform-specific Python script path
case "$PLATFORM" in
  gcp|aws|azure)
    PYTHON_SCRIPT_PATH="/app/execution/${PYTHON_SCRIPT}"
    IMAGE_FILE_PATH="${EFFECTIVE_OM_PATH}/${JOB_TITLE}_aligned.tif"
    
    # Safe basename calls with null checks
    if [ -n "$SHAPEFILE_PATH1" ]; then
        SHAPEFILE_PATH1_LOCAL="${EFFECTIVE_SHAPEFILE_PATH}/$(basename "$SHAPEFILE_PATH1")"
    else
        SHAPEFILE_PATH1_LOCAL=""
    fi
    
    if [ -n "$SHAPEFILE_PATH2" ]; then
        SHAPEFILE_PATH2_LOCAL="${EFFECTIVE_SHAPEFILE_PATH}/$(basename "$SHAPEFILE_PATH2")"
    else
        SHAPEFILE_PATH2_LOCAL=""
    fi
    ;;
  slurm)
    PYTHON_SCRIPT_PATH="./${PYTHON_SCRIPT}"
    IMAGE_FILE_PATH="${OM_ALIGNED_FOLDER}${JOB_TITLE}_aligned.tif"
    SHAPEFILE_PATH1_LOCAL="${SHAPEFILE_PATH1}"
    SHAPEFILE_PATH2_LOCAL="${SHAPEFILE_PATH2}"
    ;;
esac

echo "Using Python script: $PYTHON_SCRIPT_PATH"
echo "Image file path: $IMAGE_FILE_PATH"

conda run -n harvest python "$PYTHON_SCRIPT_PATH" \
    --csv_folder_path "$EFFECTIVE_CSV_PATH" \
    --image_folder_path "$IMAGE_FILE_PATH" \
    --shapefile_path1 "$SHAPEFILE_PATH1_LOCAL" \
    --shapefile_path2 "$SHAPEFILE_PATH2_LOCAL" \
    --output_path1 "$EFFECTIVE_OUTPUT_PATH" \
    --output_path2 "$EFFECTIVE_OUTPUT_PATH" \
    --crop1 "$CROP1" \
    --crop2 "$CROP2" \
    --location "$OM" \
    --date "$DATE" \
    --plotimage_source "$PLOTIMAGE_SOURCE"

echo "Core plot tile extraction processing finished."

# --- 4. PERFORMANCE LOGGING ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))
processing_step="plottile"

echo "Finding combined area of plot polygons"

# Initialize variables to 0 to prevent errors if the python script returns nothing
plot_area_corn_m2=0
plot_area_soy_m2=0

# Platform-specific utility script path and shapefile paths
case "$PLATFORM" in
  gcp|aws|azure)
    UTILS_SCRIPT="/app/execution/utils/get_plot_area.py"
    SHAPEFILE_CORN="$SHAPEFILE_PATH1_LOCAL"
    SHAPEFILE_SOY="$SHAPEFILE_PATH2_LOCAL"
    ;;
  slurm)
    UTILS_SCRIPT="../utils/get_plot_area.py"
    SHAPEFILE_CORN="$SHAPEFILE_PATH1"
    SHAPEFILE_SOY="$SHAPEFILE_PATH2"
    ;;
esac

# Only run the python script if the shapefile path is not an empty string
if [ -n "$SHAPEFILE_CORN" ] && [ -f "$SHAPEFILE_CORN" ]; then
    plot_area_corn_m2=$(conda run -n harvest python "$UTILS_SCRIPT" "$SHAPEFILE_CORN" 2>/dev/null || echo "0")
fi

if [ -n "$SHAPEFILE_SOY" ] && [ -f "$SHAPEFILE_SOY" ]; then
    plot_area_soy_m2=$(conda run -n harvest python "$UTILS_SCRIPT" "$SHAPEFILE_SOY" 2>/dev/null || echo "0")
fi

# Use parameter expansion to default to 0 if a variable is empty or unset
plot_area_total_m2=$(echo "${plot_area_corn_m2:-0} + ${plot_area_soy_m2:-0}" | bc -l 2>/dev/null || echo "0")

# --- 5. UPLOAD RESULTS & LOGS (Cloud platforms only) ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Uploading results to ${OUTPUT_PATH_PLOTTILES}"
    rclone copy "$EFFECTIVE_OUTPUT_PATH" "$OUTPUT_PATH_PLOTTILES" --gcs-bucket-policy-only
    
    # Create and upload performance log
    LOCAL_LOG_FILE="$LOCAL_LOG_DIR/execution_times_plottile.csv"
    
    # Create log file with header if it doesn't exist
    if [ ! -f "$LOCAL_LOG_FILE" ]; then
        echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,plot_area_total_m2,output_size" > "$LOCAL_LOG_FILE"
    fi
    
    # Get the size of the output folder
    output_size=$(du -sh "$EFFECTIVE_OUTPUT_PATH" | awk '{print $1}')
    
    # Append the data
    echo "$JOB_IDENTIFIER,$JOB_TITLE,$processing_step,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$plot_area_total_m2,$output_size" >> "$LOCAL_LOG_FILE"
    
    # Upload performance log
    echo "Uploading performance log to ${PERF_LOG_DIR}"
    rclone copy "$LOCAL_LOG_FILE" "$PERF_LOG_DIR" --gcs-bucket-policy-only
    ;;
  slurm)
    # For SLURM, write directly to the performance log
    log_file="${PERF_LOG_DIR}/execution_times_plottile.csv"
    
    # Check if the log file exists
    if [ ! -f "$log_file" ]; then
        # File doesn't exist, so write the header
        echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,plot_area_total_m2,output_size" > "$log_file"
    fi
    
    # Get the size of the output folder
    output_size=$(du -sh "$EFFECTIVE_OUTPUT_PATH" | awk '{print $1}')
    
    # Append the data
    echo "$JOB_IDENTIFIER,$JOB_TITLE,$processing_step,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$plot_area_total_m2,$output_size" >> "$log_file"
    ;;
esac

echo "Job finished successfully. Total execution time: $execution_time seconds"