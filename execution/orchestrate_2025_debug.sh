#!/bin/bash
#SBATCH --job-name=folder-watcher
#SBATCH --output=/fs/ess/PAS2699/nitrogen/data/uas/2025_test/processing/logs_folderwatcher/folder_watcher_%j.out
#SBATCH --error=/fs/ess/PAS2699/nitrogen/data/uas/2025_test/processing/logs_folderwatcher/folder_watcher_%j.err
#SBATCH --time=7-00:00:00  # 7 days
#SBATCH --ntasks=1             # Only 1 task needed
#SBATCH --cpus-per-task=1      # Just 1 CPU core
#SBATCH --mem=1G               # Very low memory usage
#SBATCH --partition=batch
#SBATCH -A PAS2699



source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest


python -u orchestrate.py \
    --config_file /fs/ess/PAS2699/nitrogen/data/uas/2025/config/uas_config.yaml \
    --dry_run

# Example of how to run the script with different arguments
#    --flight "wooster" \
#    --dry_run
#    --steps "step4" "step5"
#    --date_range "20250610" "20250610"

# If the argument is "filtered" then run the job with a different output path
if [ $# -eq 1 ] && [ "$1" == "filtered" ]; then
    echo "Running for filtered flights"
    python -u ../utils/displayjson_jobid.py
    python -u ../utils/displayjson_flightdict.py --json_file ../profiling/flight_dict_filtered.json \
                        --output_html_file '/fs/ess/PAS2699/nitrogen/data/uas/published/status/report_flight_status_filtered.html'
else
    echo "Running for all flights"
    python -u ../utils/displayjson_jobid.py --json_file ../profiling/job_id.json \
                        --output_html_file '/fs/ess/PAS2699/nitrogen/data/uas/published/status/report_job_status.html'
    python -u ../utils/displayjson_flightdict.py --json_file ../profiling/flight_dict.json \
                        --output_html_file '/fs/ess/PAS2699/nitrogen/data/uas/published/status/report_flight_status.html'
fi