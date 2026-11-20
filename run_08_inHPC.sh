#!/bin/bash

# module loads
module purge
module load release/25.06


sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=large_count_container_samples
#SBATCH --output=./logs/slurm_%a_%j.out
#SBATCH --error=./logs/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=48000
#SBATCH --time=01:00:00
#SBATCH --account=p_scads_autoencodix


source .venv/bin/activate
# Count samples per split (train, valid, test) in each acx container, loaded one at a time
python ./src/08_count_container_samples.py

exit 0
EOT
