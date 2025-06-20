#!/bin/bash
#SBATCH -J "cc_fsr_nitrogen_rgb_20250610"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_inf_cc/%j-fsr_nitrogen_rgb_20250610-inf_cc.txt
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_inf_cc/%j-fsr_nitrogen_rgb_20250610-inf_cc.err
#SBATCH --time=00:20:00
#SBATCH --mem=100G
#SBATCH -A PAS2699

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

echo "Running canopy coverage inference for: fsr_nitrogen_rgb_20250610"

num_cores=$(nproc)
echo Number of cores: $num_cores

job_name="fsr_nitrogen_rgb_20250610"

start_time=$(date +%s)

python "inference_canopy_cover.py" \
    --input_dir "/fs/ess/PAS2699/nitrogen/data/uas/2025/plottiles/plot_tiles_rgb_om/fsr_nitrogen_rgb_20250610/" \
    --output_json "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/inf_cc_fsr_nitrogen_om_20250610.json" \
    --model_path "/fs/ess/PAS2699/nitrogen/models/canopy_coverage/cc_kmeans_model.pkl" \
    --field "fsr_nitrogen" \
    --plotimage_source "om" \
    --date "20250610"

# Get the size of the output folder
output_size=$(du -sh "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/" | awk '{print $1}')

end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="canopy_coverage_inference"

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_canopy_coverage.csv"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size" > "$log_file"
fi

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"