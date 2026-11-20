#!/bin/bash

# module loads
module purge
module load release/25.06

# Retrieve chunks in parallel 
test="no"

if [ "$test" == "test" ]; then
	echo "Running in test mode"
	total_parts=12
	full_ram=12000
	reduced_ram=4000
else
	echo "Running in full mode"
	total_parts=64
	full_ram=48000
	reduced_ram=12000
fi

# Fallback: rerun only specific parts that failed (e.g. due to time-outs).
# To use it, uncomment the next two lines and comment out the full loop below.
# select_parts_list=(0 17)
# for i in "${select_parts_list[@]}";do # Rerun specific parts only
for i in $(seq 0 $total_parts);do # Parallelize on HPC for speed up (takes around 24h)
	# Set memory when i is smaller than 16 (heaviest chunks)
	if [ $i -lt 16 ]; then
		mem_per_cpu=$full_ram
	else
		mem_per_cpu=$reduced_ram
	fi
	echo "Submitting job for part $i out of $total_parts"
	sbatch <<-EOT
	#!/bin/bash
	#SBATCH --job-name=large_ontix_census
	#SBATCH --output=./logs/slurm_%a_%j.out
	#SBATCH --error=./logs/slurm_%a_%j.err
	#SBATCH --nodes=1
	#SBATCH --tasks-per-node=1
	#SBATCH --cpus-per-task=16
	#SBATCH --mem-per-cpu=$mem_per_cpu
	#SBATCH --time=48:00:00
	#SBATCH --account=p_scads_autoencodix


	source .venv/bin/activate
	# Retrieve chunks
	python ./src/01_get_census_data.py --step 2 --part $i --total-parts $total_parts --test-mode $test

	exit 0
	EOT
done
