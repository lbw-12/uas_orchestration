#!/bin/bash
#SBATCH -J "generating_geojson"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_geojson/%j.txt
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_geojson/%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=100G
#SBATCH -A PAS2699

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

echo "Running to generating geojson for: northwest_bftb_ms_20250526"

num_cores=$(nproc)
echo Number of cores: $num_cores

job_name="northwest_bftb_ms_20250526"

start_time=$(date +%s)

python "process_geojson.py" \
    --inference_folder "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/" \
    --location "northwest_bftb" \
    --date "20250526" \
    --output_json "/fs/ess/PAS2699/nitrogen/data/uas/2025/2025/northwest_bftb/20250526/northwest_bftb_20250526.geojson"

end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="generating_geojson"

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_geojson.csv"

# Get the size of the output folder
output_size=$(du -sh "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/" | awk '{print $1}')

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size" > "$log_file"
fi

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"