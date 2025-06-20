#!/bin/bash
#SBATCH -J "sr_western_ncc_ms_20250611"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_inf_sr/%j-western_ncc_ms_20250611-inf_sr.txt
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_inf_sr/%j-western_ncc_ms_20250611-inf_sr.err
#SBATCH --time=00:20:00
#SBATCH --mem=100G
#SBATCH -A PAS2699

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

echo "Running spectral reflectance inference for: western_ncc_ms_20250611"

num_cores=$(nproc)
echo Number of cores: $num_cores

job_name="western_ncc_ms_20250611"

start_time=$(date +%s)

python "inference_spectral_reflectance.py" \
    --input_dir "/fs/ess/PAS2699/nitrogen/data/uas/2025/plottiles/plot_tiles_ms_om/western_ncc_ms_20250611/" \
    --output_dir "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/" \
    --output_json "/fs/ess/PAS2699/nitrogen/data/uas/2025/inference/inf_sr_western_ncc_om_20250611.json" \
    --model_path "/fs/ess/PAS2699/nitrogen/models/spectral_reflectance/sr_rf_classifier_model.pkl" \
    --field "western_ncc" \
    --plotimage_source "om" \
    --date "20250611"


end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="spectral_reflectance_inference"

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_spectral_reflectance.csv"

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