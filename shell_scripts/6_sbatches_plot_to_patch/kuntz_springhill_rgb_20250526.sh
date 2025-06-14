#!/bin/bash
#SBATCH -J "plottile_to_patches_rgb"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_plot_to_patch/%j.txt
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_plot_to_patch/%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=10G
#SBATCH -A PAS2699

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

echo "Running plot tile to patches extraction for: kuntz_springhill_rgb_20250526"

num_cores=$(nproc)
echo Number of cores: $num_cores

job_name="kuntz_springhill_rgb_20250526"

start_time=$(date +%s)

python "plot_tiles_to_patches.py" \
    --input_dir "/fs/ess/PAS2699/nitrogen/data/uas/2025/plottiles/plot_tiles_rgb_om/kuntz_springhill_rgb_20250526/" \
    --output_dir "/fs/ess/PAS2699/nitrogen/data/uas/2025/plot_patches/kuntz_springhill_om_rgb_20250526/"


end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="plottile_to_patches_rgb"

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_plottile_to_patches_rgb.csv"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size" > "$log_file"
fi

# Get the size of the output folder
output_size=$(du -sh "/fs/ess/PAS2699/nitrogen/data/uas/2025/plot_patches/kuntz_springhill_om_rgb_20250526/" | awk '{print $1}')

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"