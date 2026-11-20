#!/bin/bash

# module loads
module purge
module load release/25.06

ont_list=("Dim16_ClaudeOpus5" "Dim32_ClaudeOpus5" "Dim48_ClaudeOpus5" "Dim16_KimiK3" "Dim32_KimiK3" "Dim48_KimiK3" "Dim16_GPT5.6_TerraPro" "Dim32_GPT5.6_TerraPro" "Dim48_GPT5.6_TerraPro")
# ont_list=("Dim48_ClaudeOpus5")

time_for_all=96  # in hours
# time_for_all=1 # in hours, test run

## Start Ontix trainings
for ont in ${ont_list[@]}; do
	echo "Training for ontology: $ont"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_ontix_full_train_Aug26
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --cpus-per-task=12
	#SBATCH --mem-per-cpu=15G
	#SBATCH --time=${time_for_all}:00:00
	#SBATCH --gres=gpu:1
	#SBATCH --account=p_scads_autoencodix

	source .venv/bin/activate
	python ./src/04_large_ontix_finaltraining.py $ont "large_ontix_${ont}_tuning.pkl"

	exit 0
	EOT
done

## Start Varix trainings

ont_list=("Dim16_Varix" "Dim32_Varix" "Dim48_Varix")
# ont_list=("Dim48_Varix")

time_for_all=96  # in hours
# time_for_all=1 # in hours, test run


for ont in ${ont_list[@]}; do
	echo "Training for ontology: $ont"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_varix_full_train_Aug26
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --cpus-per-task=12
	#SBATCH --mem-per-cpu=15G
	#SBATCH --time=${time_for_all}:00:00
	#SBATCH --gres=gpu:1
	#SBATCH --account=p_scads_autoencodix

	source .venv/bin/activate
	python ./src/04b_large_varix_finaltraining.py $ont 

	exit 0
	EOT
done

## Start Disentanglix trainings

ont_list=("Dim16_Disentanglix" "Dim32_Disentanglix" "Dim48_Disentanglix")
# ont_list=("Dim48_Disentanglix")

time_for_all=96  # in hours
# time_for_all=1 # in hours, test run

for ont in ${ont_list[@]}; do
	echo "Training for ontology: $ont"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_disentanglix_full_train_Aug26
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --cpus-per-task=12
	#SBATCH --mem-per-cpu=15G
	#SBATCH --time=${time_for_all}:00:00
	#SBATCH --gres=gpu:1
	#SBATCH --account=p_scads_autoencodix

	source .venv/bin/activate
	python ./src/04c_large_disentanglix_finaltraining.py $ont 

	exit 0
	EOT
done