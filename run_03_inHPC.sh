#!/bin/bash

# module loads
module purge
module load release/25.06


sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=large_ontix_census
#SBATCH --output=./logs/slurm_%a_%j.out
#SBATCH --error=./logs/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=48000
#SBATCH --time=03:00:00
#SBATCH --account=p_scads_autoencodix


source .venv/bin/activate
# Combine chunks and split data, save as h5ad files
python ./src/02_census_preparation.py --step 1
# Prepare acx containers for each data split
python ./src/02_census_preparation.py --step 2 --split train
python ./src/02_census_preparation.py --step 2 --split tune
python ./src/02_census_preparation.py --step 2 --split holdout

exit 0
EOT