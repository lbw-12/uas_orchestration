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


python -u orchestrate.py \
    --config_file /Users/lukewaltz/VisualStudio/uas_orchestration-dev/yaml/uas_config_gcp.yaml \
    --platform 'gcp' \
    --dry_run







    #--steps "step1" "step2" "step3" "step7" "step9" "step10" "step11" "step15" \


# Example of how to run the script with different arguments
#    --flight "wooster" \
#    --dry_run
#    --steps "step4" "step5"
#    --date_range "20250610" "20250610"
#    --file_age 1 # 1 second file age if you're moving files around for testing. Default is 3600 seconds (1 hour) to ensure any uploads are finished.


python -u ../utils/displayjson_jobid.py --json_file ../profiling/job_id.json \
                        --output_html_file '/fs/ess/PAS2699/nitrogen/data/uas/published/status/report_job_status.html'

python -u ../utils/displayjson_flightdict.py --json_file ../profiling/flight_dict.json \
                        --output_html_file '/fs/ess/PAS2699/nitrogen/data/uas/published/status/report_flight_status.html'