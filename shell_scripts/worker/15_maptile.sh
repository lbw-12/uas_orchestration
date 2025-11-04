#!/bin/bash
set -e

# --- 1. SETUP: Read environment variables & define local paths ---
echo "Starting Maptile Creation Job: ${JOB_NAME}, ID: ${JOB_ID}"
start_time=$(date +%s)

NUM_CORES=$(nproc)
echo "Number of cores: $NUM_CORES"

OM_FILENAME="${JOB_TITLE}_aligned.tif"

# --- 2. CONFIGURATION & DATA STAGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Platform: $PLATFORM. Configuring for cloud execution using rclone."
    JOB_IDENTIFIER="${JOB_ID}"
    JOB_MEMORY="${MEMORY}"
    LOCAL_OM_DIR="/data/input/orthomosaics"
    LOCAL_OUTPUT_DIR="/data/output/maptiles"
    LOCAL_LOG_DIR="/data/logs"
    mkdir -p "$LOCAL_OM_DIR" "$LOCAL_OUTPUT_DIR" "$LOCAL_LOG_DIR"
    
    EFFECTIVE_OM_DIR="$LOCAL_OM_DIR"
    EFFECTIVE_OUTPUT_DIR="$LOCAL_OUTPUT_DIR"
    INPUT_ORTHOMOSAIC="$EFFECTIVE_OM_DIR/$OM_FILENAME"

    echo "Downloading aligned orthomosaic from ${OM_ALIGNED_FOLDER}${OM_FILENAME}"
    rclone copy "${OM_ALIGNED_FOLDER}${OM_FILENAME}" "$EFFECTIVE_OM_DIR/" --gcs-bucket-policy-only
    ;;
  slurm)
    echo "Platform: slurm. Configuring for local execution."
    JOB_IDENTIFIER="${SLURM_JOB_ID}"
    JOB_MEMORY="${SLURM_MEM_PER_NODE}"
    EFFECTIVE_OM_DIR="$OM_ALIGNED_FOLDER"
    EFFECTIVE_OUTPUT_DIR="$MAPTILES_DIR"
    INPUT_ORTHOMOSAIC="$EFFECTIVE_OM_DIR/$OM_FILENAME"
    mkdir -p "$EFFECTIVE_OUTPUT_DIR"
    ;;
  *)
    echo "Error: Unsupported PLATFORM value '${PLATFORM}'." >&2
    exit 1
    ;;
esac

echo "Job Identifier: ${JOB_IDENTIFIER}"
free -h

# --- 3. CORE LOGIC: Generate maptiles using gdal2tiles ---
echo "Starting maptile generation"
echo "Input orthomosaic: ${INPUT_ORTHOMOSAIC}, Output: ${EFFECTIVE_OUTPUT_DIR}"

if [ ! -f "$INPUT_ORTHOMOSAIC" ]; then
    echo "Error: Input orthomosaic not found at $INPUT_ORTHOMOSAIC"
    exit 1
fi

# Set the desired zoom levels
ZOOM_LEVELS="15-24"
PROFILE="mercator"
RESAMPLING="average"
NB_PROCESSES=$NUM_CORES

echo "Starting tile generation for zoom levels $ZOOM_LEVELS..."
echo "Number of processes for gdal2tiles: $NB_PROCESSES"

# Try to find gdal2tiles (with or without .py extension)
if command -v gdal2tiles.py &> /dev/null; then
    GDAL2TILES_EXEC="gdal2tiles.py"
elif command -v gdal2tiles &> /dev/null; then
    GDAL2TILES_EXEC="gdal2tiles"
else
    echo "Error: 'gdal2tiles' or 'gdal2tiles.py' command not found. Make sure GDAL is installed and your conda environment is activated."
    exit 1
fi

echo "Using GDAL2TILES executable: $GDAL2TILES_EXEC"

GDAL2TILES_CMD="${GDAL2TILES_EXEC} --zoom=${ZOOM_LEVELS} --profile=${PROFILE} --resampling=${RESAMPLING} --processes=${NB_PROCESSES} --webviewer=none"
GDAL2TILES_CMD="${GDAL2TILES_CMD} \"${INPUT_ORTHOMOSAIC}\" \"${EFFECTIVE_OUTPUT_DIR}\""

echo "Running command: $GDAL2TILES_CMD"
eval $GDAL2TILES_CMD

if [ $? -eq 0 ]; then
    echo "Tile generation completed successfully!"
else
    echo "Error: Tile generation failed."
    exit 1
fi

# --- 4. PERFORMANCE LOGGING ---
end_time=$(date +%s)
execution_time=$((end_time - start_time))
processing_step="maptile"

echo "Finding area of orthomosaic in square meters"

# Platform-specific utils script path
case "$PLATFORM" in
  gcp|aws|azure)
    UTILS_SCRIPT="/app/utils/get_om_area.py"
    ;;
  slurm)
    UTILS_SCRIPT="../utils/get_om_area.py"
    ;;
esac

area_m2=$(conda run -n harvest python "$UTILS_SCRIPT" --ortho "$INPUT_ORTHOMOSAIC" --unit "m2")

# Get the size of the output folder
output_size=$(du -sh "$EFFECTIVE_OUTPUT_DIR" | awk '{print $1}')

data_line="$JOB_IDENTIFIER,$JOB_TITLE,$processing_step,$NUM_CORES,$JOB_MEMORY,$start_time,$end_time,$execution_time,$area_m2,$output_size"

# --- 5. STAGE-OUT & LOGGING ---
case "$PLATFORM" in
  gcp|aws|azure)
    echo "Uploading maptiles to ${MAPTILES_DIR}"
    rclone copy "$EFFECTIVE_OUTPUT_DIR" "$MAPTILES_DIR" --gcs-bucket-policy-only
    
    log_file="$LOCAL_LOG_DIR/perf_${JOB_IDENTIFIER}.csv"
    header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,area_m2,output_size"
    echo "$header" > "$log_file"
    echo "$data_line" >> "$log_file"
    echo "Uploading performance log to ${PERF_LOG_DIR}"
    rclone copy "$log_file" "$PERF_LOG_DIR" --gcs-bucket-policy-only
    ;;
  slurm)
    log_file="$PERF_LOG_DIR/execution_times_maptile.csv"
    if [ ! -f "$log_file" ]; then
      header="job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,area_m2,output_size"
      echo "$header" > "$log_file"
    fi
    echo "$data_line" >> "$log_file"
    echo "Performance log updated at $log_file"
    
    # For SLURM, also run folders_to_json.py
    maptiles_base_dir="${MAPTILES_DIR%/*/*/*}"
    FOLDERS_TO_JSON_SCRIPT="../utils/folders_to_json.py"
    if [ -f "$FOLDERS_TO_JSON_SCRIPT" ]; then
        conda run -n harvest python "$FOLDERS_TO_JSON_SCRIPT" --scan_dir "$maptiles_base_dir"
    fi
    ;;
esac

echo "Job finished successfully. Total execution time: $execution_time seconds"
echo "Tiles are located in: $EFFECTIVE_OUTPUT_DIR"

