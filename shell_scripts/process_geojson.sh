#!/bin/bash
#SBATCH -J "generating_geojson"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/%j-logs_geojson/process_geojson.txt
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/%j-logs_geojson/process_geojson.err
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH -A PAS2699

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

echo "Generating GeoJSON"



python -u "process_geojson_batch.py" \
    --config_file "/fs/ess/PAS2699/nitrogen/data/uas/2025/config/uas_config.yaml"

python -u ../utils/folders_to_json.py --scan_dir "/fs/ess/PAS2699/nitrogen/data/uas/published/osu_public"
python -u ../utils/folders_to_json.py --scan_dir "/fs/ess/PAS2699/nitrogen/data/uas/published/bergman_private"
python -u ../utils/folders_to_json.py --scan_dir "/fs/ess/PAS2699/nitrogen/data/uas/published/douglass_private"
python -u ../utils/folders_to_json.py --scan_dir "/fs/ess/PAS2699/nitrogen/data/uas/published/frantom_private"
python -u ../utils/folders_to_json.py --scan_dir "/fs/ess/PAS2699/nitrogen/data/uas/published/kuntz_private"
