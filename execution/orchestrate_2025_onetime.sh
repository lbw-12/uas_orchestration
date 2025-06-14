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
    --regen_shell_scripts \
    --dry_run \
    --steps "step6" "step7" "step8" "step9" "step10" "step11"


    
#    --location "wooster" \
#    --dry_run
#    --steps "step4" "step5"
#    --regen_shell_scripts

