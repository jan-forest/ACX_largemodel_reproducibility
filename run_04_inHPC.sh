#!/bin/bash

# module loads
module purge
module load release/25.06

ont_list=("Dim16_ClaudeOpus5" "Dim32_ClaudeOpus5" "Dim48_ClaudeOpus5" "Dim16_KimiK3" "Dim32_KimiK3" "Dim48_KimiK3" "Dim16_GPT5.6_TerraPro" "Dim32_GPT5.6_TerraPro" "Dim48_GPT5.6_TerraPro")
# ont_list=("Dim48_ClaudeOpus5")

n_workers=4
# n_workers=1
time_for_tuning=48  # in hours
# time_for_tuning=1
time_for_all=$((time_for_tuning+1))  # in hours
# time_for_all=1

for ont in ${ont_list[@]}; do
	echo "Tuning for ontology: $ont"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_ontix_tune_Aug26
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=3
#	#SBATCH --nodes=1
	#SBATCH --mincpus=1
	#SBATCH --ntasks-per-node=3
#	#SBATCH --ntasks-per-node=1
	#SBATCH --cpus-per-task=6
	#SBATCH --mem-per-cpu=20G
	#SBATCH --time=${time_for_all}:00:00
	#SBATCH --gres=gpu:${n_workers}
	#SBATCH --account=p_scads_autoencodix

	source .venv/bin/activate
	python ./src/03_large_ontix_tuning.py $ont $time_for_tuning $n_workers

	exit 0
	EOT
done