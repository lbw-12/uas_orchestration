#!/bin/bash
#SBATCH --job-name=vitr26_vittransform
#SBATCH --output=vitr26_vittransform_%j.log
#SBATCH --error=vitr26_vittransform_%j.err
#SBATCH --time=15:00:00  # Set the job time limit
#SBATCH --mem=300G  # Amount of memory
#SBATCH --gpus-per-node=1  # Number of GPUs per node
#SBATCH -A PAS2699
#SBATCH --mail-type=ALL
#SBATCH --mail-user="sridhar.86@buckeyemail.osu.edu"

module load python
source activate agri
pip install wandb==0.15.11
pip install timm
#pip install optuna
python main.py
#python tuner.py