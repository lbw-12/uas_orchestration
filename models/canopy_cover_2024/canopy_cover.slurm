#!/bin/sh
#SBATCH --job-name=kmeans_1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user="sridhar.86@buckeyemail.osu.edu"
#SBATCH --time=10:00:00
#SBATCH --account=PAS2699
#SBATCH --output=kmeans_1000.out.%j
#SBATCH --mem-per-cpu=100G

cd /fs/ess/PAS0272/sarikaa/canopy_cover/minibatch_kmeans
module load python
source activate agri
pip install colorspacious
python canopy_pipeline.py --input_dir /fs/ess/PAS2699/nitrogen/data/uas/2024/plottiles/plot_tiles_rgb_om --output_dir outputs --model_path "minibatch_kmeans_model_1000.pkl" --train --n_clusters 2 --batch_size 100000 --visualize