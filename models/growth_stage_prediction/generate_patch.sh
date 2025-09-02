#!/bin/bash
#SBATCH --job-name=patch_extraction-om
#SBATCH --output=patch_extraction_om_%j.log
#SBATCH --error=patch_extraction_om_%j.err
#SBATCH --time=05:00:00  # Set the job time limit
#SBATCH --mem=10G  # Amount of memory
#SBATCH -A PAS2699
#SBATCH --mail-type=ALL
#SBATCH --mail-user="sridhar.86@buckeyemail.osu.edu"

module load python
source activate agri
# CONFIGURATION
CONFIG_FILE="/fs/ess/PAS2699/nitrogen/data/uas/2024/config/uas_config.yaml"
FLAG="om"  # or "ir" 

# LOAD CONFIG VALUES
BASE_FOLDER=$(yq '.base_folder' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')
PLOTTILES_FOLDER=$(yq '.plottiles_folder' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')
PLOTTILES_FOLDER_RGB=$(yq -r '.plottiles_om.rgb' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')
PLOTTILES_FOLDER_MS=$(yq -r '.plottiles_om.ms' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')
PLOTTILES_PATH_RGB="${BASE_FOLDER}${PLOTTILES_FOLDER}${PLOTTILES_FOLDER_RGB}"
PLOTTILES_PATH_MS="${BASE_FOLDER}${PLOTTILES_FOLDER}${PLOTTILES_FOLDER_MS}"
PATCHES_FOLDER=$(yq '.patch_folder' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')
PATCHES_FOLDER_RGB=$(yq -r '.patches.rgb' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')
PATCHES_FOLDER_MS=$(yq -r '.patches.ms' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')
PATCHES_PATH_RGB="${BASE_FOLDER}${PATCHES_FOLDER}${PATCHES_FOLDER_RGB}"
PATCHES_PATH_MS="${BASE_FOLDER}${PATCHES_FOLDER}${PATCHES_FOLDER_MS}"
export PATCH_TO_PLOT_TILES_SCRIPT="/fs/ess/PAS0272/sarikaa/growth_stage_prediction/plot_tiles_to_patches.py"                                      #$(yq '.uas_pipeline.plot_tiles' "$CONFIG_FILE"| sed 's/^"\(.*\)"$/\1/')

if [[ "$FLAG" == "om" ]]; then
    echo "Processing patches for OM data..."
    for FILE in "$PLOTTILES_PATH_RGB"/*; do
        filename=$(basename "$FILE")
        echo "Processing folder: ${filename}, PLOTTILES_PATH: $PLOTTILES_PATH_RGB, PATCHES_PATH: $PATCHES_PATH_RGB"
        python "$PATCH_TO_PLOT_TILES_SCRIPT" \
            --plot_filename "${PLOTTILES_PATH_RGB}/${filename}" \
            --patch_size 224 \
            --output_path "${PATCHES_PATH_RGB}" \

    done
fi