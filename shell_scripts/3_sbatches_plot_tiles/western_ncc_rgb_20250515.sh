#!/bin/bash
#SBATCH -J "plottile_western_ncc_rgb_20250515"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_plot_tiles/%j.txt
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_plot_tiles/%j.err
#SBATCH --time=0:20:00
#SBATCH --mem=4G
#SBATCH -A PAS2699

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

echo "Running plot tile extraction for: western_ncc_rgb_20250515"

num_cores=$(nproc)
echo Number of cores: $num_cores

job_name="western_ncc_rgb_20250515"

start_time=$(date +%s)

python "plot_tiles_ir_om.py" \
    --csv_file_path "" \
    --image_folder_path "/fs/ess/PAS2699/nitrogen/data/uas/2025/orthomosaics/aligned/western_ncc_rgb_20250515_aligned.tif" \
    --shapefile_path_soy "" \
    --shapefile_path_corn "/fs/ess/PAS2699/nitrogen/data/uas/2025/shapefiles/plots/western_n_cc/western_n_cc.shp" \
    --output_path_soy "/fs/ess/PAS2699/nitrogen/data/uas/2025/plottiles/plot_tiles_rgb_om/western_ncc_rgb_20250515/" \
    --output_path_corn "/fs/ess/PAS2699/nitrogen/data/uas/2025/plottiles/plot_tiles_rgb_om/western_ncc_rgb_20250515/" \
    --location "western_ncc" \
    --date "20250515" \
    --plotimage_source "om"


end_time=$(date +%s)
execution_time=$((end_time - start_time))

processing_step="plottile"

echo "finding combined area of plot polygons"

plot_area_corn_m2=$(python -u ../utils/get_plot_area.py "/fs/ess/PAS2699/nitrogen/data/uas/2025/shapefiles/plots/western_n_cc/western_n_cc.shp")
plot_area_soy_m2=$(python -u ../utils/get_plot_area.py "")
plot_area_total_m2=$(echo "$plot_area_corn_m2 + $plot_area_soy_m2" | bc)

log_dir=/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf
log_file="$log_dir/execution_times_plottile.csv"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
  # File doesn't exist, so write the header
  echo "job_id,job_name,processing_step,num_cores,memory,start_time,end_time,execution_time,plot_area_total_m2,output_size" > "$log_file"
fi

# Get the size of the output folder
output_size=$(du -sh "/fs/ess/PAS2699/nitrogen/data/uas/2025/plottiles/plot_tiles_rgb_om/western_ncc_rgb_20250515/" | awk '{print $1}')

# Append the data
echo "$SLURM_JOB_ID,$job_name,$processing_step,$num_cores,$SLURM_MEM_PER_NODE,$start_time,$end_time,$execution_time,$plot_area_total_m2,$output_size" >> "$log_file"

echo "All jobs are done, total time of execution: $execution_time seconds"