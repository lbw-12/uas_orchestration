#!/bin/bash
#SBATCH -J "maptile_western_nsc_rgb_20250526"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -A PAS2699
#SBATCH -o "/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_maptiles/%j.txt"
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_maptiles/%j.err
#SBATCH --mem=32GB
#SBATCH --time=01:00:00

# Ensure log directory exists
mkdir -p "/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_maptiles"

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

num_cores=$(nproc)
echo Number of cores: $num_cores

start_time=$(date +%s)

job_name="western_nsc_rgb_20250526"

OM_DIR="/fs/ess/PAS2699/nitrogen/data/uas/2025/orthomosaics/aligned/"
OM_FILENAME="western_nsc_rgb_20250526_aligned.tif"


echo "Running map tile creation for: western_nsc_rgb_20250526"


# Set the path to your input orthomosaic raster file (use OSC project/scratch paths)
INPUT_ORTHOMOSAIC="$OM_DIR/$OM_FILENAME"

# Set the desired output directory for the tiles

# Create maptile directory exists
mkdir -p "/fs/ess/PAS2699/nitrogen/data/uas/published/osu_public/2025/western_nsc/20250526"
OUTPUT_DIRECTORY="/fs/ess/PAS2699/nitrogen/data/uas/published/osu_public/2025/western_nsc/20250526"

# Set the desired zoom levels
ZOOM_LEVELS="15-24"

# --- Optional gdal2tiles.py Parameters ---
PROFILE="mercator"
RESAMPLING="average"

# Use the number of cores allocated by Slurm for gdal2tiles.py
# It's good practice to use $SLURM_NTASKS or $SLURM_CPUS_ON_NODE if defined,
# or the value you set in --ntasks-per-node.
NB_PROCESSES=${SLURM_CPUS_PER_TASK:-12}  # Use 12 as default if SLURM_CPUS_PER_TASK is not set

# --- Script Logic ---

# Check if the input orthomosaic file exists
if [ ! -f "$INPUT_ORTHOMOSAIC" ]; then
    echo "Error: Input orthomosaic file not found at $INPUT_ORTHOMOSAIC"
    exit 1
fi

# Create the output directory if it doesn't exist (Slurm might run this multiple times if array job)
# For a single job, this is fine.
mkdir -p "$OUTPUT_DIRECTORY"

echo "Starting tile generation for zoom levels $ZOOM_LEVELS..."
echo "Input orthomosaic: $INPUT_ORTHOMOSAIC"
echo "Output directory: $OUTPUT_DIRECTORY"
echo "Number of processes for gdal2tiles: $NB_PROCESSES"

GDAL2TILES_EXEC="gdal2tiles"

# CHANGE 2: This is the modern, robust way to check if a command exists in your PATH.
# The 'command -v' command searches the PATH for an executable.
if ! command -v "$GDAL2TILES_EXEC" &> /dev/null; then
    # The error message is now more helpful.
    echo "Error: '$GDAL2TILES_EXEC' command not found. Make sure GDAL is installed and your conda environment is activated."
    exit 1
fi

# No other changes needed below this line. The rest of the script will work perfectly.
GDAL2TILES_CMD="${GDAL2TILES_EXEC} --zoom=${ZOOM_LEVELS} --profile=${PROFILE} --resampling=${RESAMPLING} --processes=${NB_PROCESSES} --webviewer=none"

# Add input and output file paths
GDAL2TILES_CMD="${GDAL2TILES_CMD} \"${INPUT_ORTHOMOSAIC}\" \"${OUTPUT_DIRECTORY}\""

# Execute the command
echo "Running command: $GDAL2TILES_CMD"
eval $GDAL2TILES_CMD

# Check the exit status of gdal2tiles.py
if [ $? -eq 0 ]; then
    echo "Tile generation completed successfully!"
else
    echo "Error: Tile generation failed."
    # Consider adding more specific error handling or cleanup if needed
    exit 1
fi

end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="maptile"

echo "finding area of orthomosaic in square meters"

area_m2=$(python -u ../utils/get_om_area.py --ortho "$OM_DIR/$OM_FILENAME" --unit "m2")

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_maptile.csv"

# Get the size of the output folder
output_size=$(du -sh "/fs/ess/PAS2699/nitrogen/data/uas/published/osu_public/2025/western_nsc/20250526" | awk '{print $1}')

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,area_m2,output_size" > "$log_file"
fi

# Use dirname to get the parent directory
maptiles_base_dir=$(dirname "/fs/ess/PAS2699/nitrogen/data/uas/published/osu_public/2025/western_nsc/20250526")

# execute folders_to_json.py
python -u ../utils/folders_to_json.py --scan_dir "$maptiles_base_dir"

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$area_m2,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"

echo "Tiles are located in: $OUTPUT_DIRECTORY"
echo "Job finished at $(date)"