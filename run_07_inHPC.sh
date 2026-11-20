#!/bin/bash

# module loads
module purge
module load release/25.06


time_for_all=1  # in hours

ont_list=("Dim16_ClaudeOpus5" "Dim32_ClaudeOpus5" "Dim48_ClaudeOpus5" "Dim16_KimiK3" "Dim32_KimiK3" "Dim48_KimiK3" "Dim16_GPT5.6_TerraPro" "Dim32_GPT5.6_TerraPro" "Dim48_GPT5.6_TerraPro")


for ont in ${ont_list[@]}; do
	echo "Generating data for ontology: $ont"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_ontix_generate
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --cpus-per-task=12
	#SBATCH --mem-per-cpu=5G
	#SBATCH --time=${time_for_all}:00:00
	#SBATCH --gres=gpu:1
	#SBATCH --account=p_scads_autoencodix

	source .venv/bin/activate
	python ./src/07_generate_with_ontix.py $ont

	exit 0
	EOT
done


ont_list=("Dim16_Varix" "Dim32_Varix" "Dim48_Varix")


for ont in ${ont_list[@]}; do
	echo "Generating data for varix: $ont"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_varix_generate
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --cpus-per-task=12
	#SBATCH --mem-per-cpu=5G
	#SBATCH --time=${time_for_all}:00:00
	#SBATCH --gres=gpu:1
	#SBATCH --account=p_scads_autoencodix

	source .venv/bin/activate
	python ./src/07b_generate_with_varix.py $ont

	exit 0
	EOT
done
