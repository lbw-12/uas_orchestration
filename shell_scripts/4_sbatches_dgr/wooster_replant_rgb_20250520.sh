#!/bin/bash
#SBATCH -J "dgr_wooster_replant_rgb_20250520"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0 # This gets the full memory of the node and stores it in SLURM_MEM_PER_NODE
#SBATCH -A PAS2699
#SBATCH -o "/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_dgr/%j-wooster_replant_rgb_20250520-dgr.txt"
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_dgr/%j-wooster_replant_rgb_20250520-dgr.err
#SBATCH --time=0:30:00

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

FLIGHT_DIR="/fs/ess/PAS2699/nitrogen/data/uas/2025/flights/20250520-Wooster/20250520_Wooster_Replant_Sony150 Flight 01/01_Images/20250520_Wooster_Replant_Sony150 Flight 01/OUTPUT/"
OUTPUT_PATH_GEO="/fs/ess/PAS2699/nitrogen/data/uas/2025/dgr/wooster_replant_rgb_20250520/"

job_name="wooster_replant_rgb_20250520"

num_cores=${SLURM_CPUS_ON_NODE:-$(nproc)}

# Try SLURM-provided memory, else fallback to system memory
if [ -n "$SLURM_MEM_PER_NODE" ]; then
    total_mem_mb=$SLURM_MEM_PER_NODE
elif [ -n "$SLURM_MEM_PER_CPU" ] && [ -n "$SLURM_CPUS_ON_NODE" ]; then
    total_mem_mb=$((SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE))
else
    total_mem_mb=$(free -m | awk '/^Mem:/{print $2}')
fi

echo "Running script with $num_cores cores and $total_mem_mb MB total memory"

start_time=$(date +%s)



python -u dgr_parallel.py \
    --input_folder "$FLIGHT_DIR" \
    --output_folder "$OUTPUT_PATH_GEO" \
    --num_workers $SLURM_CPUS_ON_NODE

end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="dgr"

echo "finding number of DGR images that have been processed."

image_count=$(python -u ../utils/get_imagecount.py "$OUTPUT_PATH_GEO")

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_dgr.csv"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,dgr_images,output_size" > "$log_file"
fi

# Get the size of the output folder
output_size=$(du -sh "$OUTPUT_PATH_GEO" | awk '{print $1}')

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$total_mem_mb,$start_time,$end_time,$execution_time,$image_count,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"