#!/bin/bash
#SBATCH -J "ir_frantom_rgb_20240628"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0 # This gets the full memory of the node and stores it in SLURM_MEM_PER_NODE
#SBATCH -A PAS2699
#SBATCH -o "/fs/ess/PAS2699/nitrogen/data/uas/2024/processing/logs_ir/%j.txt"
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2024/processing/logs_ir/%j.err
#SBATCH --time=0:30:00

start_time=$(date +%s)

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

OUTPUT_PATH_GEO="/fs/ess/PAS2699/nitrogen/data/uas/2024/dgr/frantom_rgb_20240628/"  
OUTPUT_PATH_IR="/fs/ess/PAS2699/nitrogen/data/uas/2024/ir/frantom_rgb_20240628/" 
ORTHO_PATH="/fs/ess/PAS2699/nitrogen/data/uas/2024/orthomosaics/v2_5_aligned/frantom_rgb_20240628_aligned.tif"

echo "------------ paths: ---------------------------------------"
echo $OUTPUT_PATH_GEO
echo $OUTPUT_PATH_IR
echo $ORTHO_PATH
echo " ----------------------------------------------------------"

num_cores=${SLURM_CPUS_ON_NODE:-$(nproc)}

# Try SLURM-provided memory, else fallback to system memory
if [ -n "$SLURM_MEM_PER_NODE" ]; then
    total_mem_mb=$SLURM_MEM_PER_NODE
elif [ -n "$SLURM_MEM_PER_CPU" ] && [ -n "$SLURM_CPUS_ON_NODE" ]; then
    total_mem_mb=$((SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE))
else
    total_mem_mb=$(free -m | awk '/^Mem:/{print $2}')
fi


echo "Running ir_parallel.py with $num_cores cores and $total_mem_mb MB total memory"

python -u ir_parallel.py "$OUTPUT_PATH_GEO" "$OUTPUT_PATH_IR" "$ORTHO_PATH" --num_workers $num_cores

end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="ir"

image_count_geo=$(python -u ../utils/get_imagecount.py "$OUTPUT_PATH_GEO")
image_count_ir=$(python -u ../utils/get_imagecount.py "$OUTPUT_PATH_IR")

# Get the size of the output folder
output_size=$(du -sh "$OUTPUT_PATH_IR" | awk '{print $1}')

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2024/processing/logs_perf
log_file="$log_dir/execution_times_ir.csv"

job_name="frantom_rgb_20240628"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,dgr_images,ir_images,output_size" > "$log_file"
fi

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$total_mem_mb,$start_time,$end_time,$execution_time,$image_count_geo,$image_count_ir,$output_size" >> "$log_file"


echo "Img Reg job done for 20240628-frantom_rgb_20240628, total time of execution: $execution_time seconds"
echo "Total input images: $image_count_geo, total output images: $image_count_ir"
