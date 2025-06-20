#!/bin/bash
#SBATCH -J "gs_northwest_bftb_rgb_20250512"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_inf_gs/%j-northwest_bftb_rgb_20250512-inf_gs.txt
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_inf_gs/%j-northwest_bftb_rgb_20250512-inf_gs.err
#SBATCH --time=00:20:00
#SBATCH --mem=300G
#SBATCH --gpus-per-node=1
#SBATCH -A PAS2699

module load cuda/12.3.0

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest




echo "Running growth stage inference for: northwest_bftb_rgb_20250512"

num_cores=$(nproc)
echo Number of cores: $num_cores

job_name="northwest_bftb_rgb_20250512"

start_time=$(date +%s)

python "inference_growth_stage.py" \
    --input_dir "/fs/ess/PAS2699/nitrogen/data/uas/2025/plot_patches/northwest_bftb_om_rgb_20250512/" \
    --output_dir "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/inf_gs_northwest_bftb_om_20250512.json" \
    --model_path "/fs/ess/PAS2699/nitrogen/models/growth_stage/gs_vit_model.pth" \
    --field "northwest_bftb" \
    --plotimage_source "om" \
    --date "20250512"



end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="growth_stage_inference"

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_growth_stage_inference.csv"

# Get the size of the output folder
output_size=$(du -sh "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/inf_gs_northwest_bftb_om_20250512.json" | awk '{print $1}')

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,output_size" > "$log_file"
fi

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"