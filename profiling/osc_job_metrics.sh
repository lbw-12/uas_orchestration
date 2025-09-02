#!/bin/bash

# This script is used to get the metrics for the jobs in the CSV file
# It uses the seff command to get the metrics
# It writes the metrics to a new CSV file
# It takes the input CSV file and the output CSV file as arguments
# It takes the job ID column name as an argument
# It takes the skip header check as an argument

# Get the input CSV file
input_csv='/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf/execution_times_maptile_cleaned.csv'
output_csv='/fs/ess/PAS2699/nitrogen/data/uas/2025/processing/logs_perf/execution_times_maptile_cleaned_aug.csv'
job_id_column='job_id'

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

python -u osc_job_metrics.py $input_csv $output_csv $job_id_column


