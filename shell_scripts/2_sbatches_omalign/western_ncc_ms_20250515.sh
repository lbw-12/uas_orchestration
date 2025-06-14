#!/bin/bash
#SBATCH -J "omalign_western_ncc_ms_20250515"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A PAS2699
#SBATCH -o "/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_omalign/%j.txt"
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_omalign/%j.err
#SBATCH --mem=128GB
#SBATCH --time=0:30:00


source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

num_cores=$(nproc)
echo Number of cores: $num_cores

start_time=$(date +%s)

job_name="western_ncc_ms_20250515"

SEARCH="western_ncc_ms_20250515.tif"
PATTERN="western_ncc_ms"
OM_FOLDER=/fs/ess/PAS2699/nitrogen/data/uas/2025/orthomosaics/initial/
OM_ALIGNED_FOLDER=/fs/ess/PAS2699/nitrogen/data/uas/2025/orthomosaics/aligned/
SHAPEFILE_PATH=/fs/ess/PAS2699/nitrogen/data/uas/2025/shapefiles/alignment/

echo "running script for omalign_western_ncc_ms_20250515"

python -u om_alignment.py \
    --search "$SEARCH" \
    --pattern "$PATTERN" \
    --shapefile_path "$SHAPEFILE_PATH" \
    --om_folder "$OM_FOLDER" \
    --om_aligned_folder="$OM_ALIGNED_FOLDER"



end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="omalign"

echo "finding number of alignment points"

alignment_points=$(python -u ../utils/get_alignment_points.py "${SHAPEFILE_PATH}/western_ncc_pts/western_ncc_pts.shp")

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_omalign.csv"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,alignment_points,output_size" > "$log_file"
fi

output_size=$(du -sh "$OM_ALIGNED_FOLDER/${job_name}_aligned.tif" | awk '{print $1}')

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$alignment_points,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"
