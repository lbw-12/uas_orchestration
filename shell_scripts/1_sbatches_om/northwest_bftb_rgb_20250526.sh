#!/bin/bash
#SBATCH -N 1
#SBATCH -A PAS2699
#SBATCH -J "om_northwest_bftb_rgb_20250526"
#SBATCH -o "/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_om/%j-northwest_bftb_rgb_20250526-om.txt"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=8:00:00
#SBATCH --mem=128G

export MPLBACKEND=Agg

# Define the directories
num_cores=$(nproc)
echo Number of cores: $num_cores
free -h

start_time=$(date +%s)

job_name="northwest_bftb_rgb_20250526"
img_dir="/fs/ess/PAS2699/nitrogen/data/uas/2025/flights/20250526-Northwest/20250526_Northwest_sony150 Flight 01/01_Images/20250526_Northwest_sony150 Flight 01/OUTPUT_northwest_bftb/"
processing_subdir="/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/northwest_bftb_rgb_20250526_$SLURM_JOB_ID"
om_dir="/fs/ess/PAS2699/nitrogen/data/uas/2025/orthomosaics/initial/"

/users/PAS2312/lwaltz/code/uas_orchestration/execution/single_job_rgb.sh "$job" "$img_dir" "$processing_subdir" $num_cores

om="northwest_bftb"
sensor_type="rgb"
date="20250526"

om_filename="northwest_bftb_rgb_20250526.tif"

# Move and rename the output file
if [ -f "$processing_subdir/code/odm_orthophoto/odm_orthophoto.tif" ]; then
    mv "$processing_subdir/code/odm_orthophoto/odm_orthophoto.tif" "$om_dir/$om_filename"
    echo "Orthomosaic moved and renamed to $om_filename."
else
    echo "Error: Orthomosaic file not found!"
fi

end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="om"

echo "Processing completed."
echo "Total execution time: $execution_time seconds"

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

echo "finding area of orthomosaic in square meters"

area_m2=$(python -u ../utils/get_om_area.py --ortho "$om_dir/$om_filename" --unit "m2")

output_size=$(du -sh "$om_dir/$om_filename" | awk '{print $1}')

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_om.csv"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,area_m2,output_size" > "$log_file"
fi

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$area_m2,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"