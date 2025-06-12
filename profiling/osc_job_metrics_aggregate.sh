#!/bin/bash

# This script is used to get the metrics for the jobs in the CSV file
# It uses the seff command to get the metrics
# It writes the metrics to a new CSV file
# It takes the input CSV file and the output CSV file as arguments
# It takes the job ID column name as an argument
# It takes the skip header check as an argument

# Get the input CSV file
input_folder='/fs/ess/PAS2699/nitrogen/data/uas/2023/processing/logs_perf/'
output_csv='/fs/ess/PAS2699/nitrogen/data/uas/2023/processing/logs_perf/aggregated_metrics_2023.csv'

source /fs/ess/PAS2699/envs/miniconda3/etc/profile.d/conda.sh
conda activate harvest

python -u osc_job_metrics_aggregate.py $input_folder $output_csv