#!/bin/bash

# module loads
module purge
module load release/25.06


sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=create_llm_ontologies
#SBATCH --output=./logs/slurm_%a_%j.out
#SBATCH --error=./logs/slurm_%a_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=p_scads_autoencodix

source .venv/bin/activate
# Create prompts
python ./src/00_llm_ontologies.py --step 1
# Query LLMs
python ./src/00_llm_ontologies.py --step 2
# Add gene id to ontologies
python ./src/00_llm_ontologies.py --step 3
# Make final ontology files
python ./src/00_llm_ontologies.py --step 4

exit 0
EOT