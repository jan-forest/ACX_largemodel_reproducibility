"""Download and chunk CELLxGENE census data for large-model training.

Each step of the pipeline can be run individually via the ``--step``
command line argument, e.g.::

	python 01_get_census_data.py --step 1
	python 01_get_census_data.py --step 2 --part 0 --total-parts 12 --test-mode test
	python 01_get_census_data.py --step all

The ``--part``/``--total-parts`` options control parallelization over chunks
(e.g. for launching several HPC jobs), and ``--test-mode`` enables a fast
test run that only processes the last 48 chunks.
"""
import argparse
import os

# Get from environment variable if set, else use current working directory
root_folder = os.getenv("WORKSPACE_FOLDER") or "."

data_folder = os.path.join(root_folder, "data/census_chunks/")

llm_ontology_folder = os.path.join(root_folder, "data/llm_ontologies/final_ontologies/")


max_chunk_size = 1000 # Downsample chunks larger than this size
perc_genes = 0.05 # Filter cells with lower than this percentage of genes expressed

# Create folder if it doesn't exist
if not os.path.exists(data_folder):
	os.makedirs(data_folder)


def step1():
	"""STEP 1 - Determine census chunks.

	Check if human_meta_counts_healthy-diseased.pkl already exists.
	"""
	print("STEP 1 - Determine census chunks")

	import cellxgene_census
	census = cellxgene_census.open_soma(census_version = "2025-11-08")

	value_filter="is_primary_data == True" # Only original data publications to avoid duplicates
	species = "homo_sapiens"

	column_levels = ["tissue_general", "cell_type_ontology_term_id", "sex", "disease"] # Will define chunks

	print("Retrieving metadata")
	## Retrieve metadata
	human_meta = (
		census["census_data"][species]
		.obs.read(column_names=column_levels, value_filter=value_filter)
		.concat()
		.to_pandas()
	)
	census.close()

	## Combine all non-healthy disease states into a single category
	human_meta["healthy"] = human_meta["disease"].apply(lambda x: "healthy" if x == "normal" else "diseased")
	human_meta = human_meta[["tissue_general", "cell_type_ontology_term_id", "sex", "healthy"]]

	print("Sample counts for top chunks:")
	print(human_meta.value_counts().head(10))
	print("Number of chunks:", human_meta.value_counts().shape[0])

	## Keep only where value count is larger than 1000
	human_meta_counts = human_meta.value_counts()
	human_meta_counts = human_meta_counts[human_meta_counts > 1000]
	print("Number of chunks with more than 1000 samples:", human_meta_counts.shape[0])
	print("Total number of samples in these chunks:", human_meta_counts.sum())

	## Save human_meta_counts 
	human_meta_counts.to_pickle(os.path.join(data_folder, "human_meta_counts_healthy-diseased.pkl"))


def step2(part: int, test_mode: str, total_parts: int):
	"""STEP 2 - Get census data chunks.

	Args:
		part: Index of the part to process (for parallelization over chunks).
		test_mode: "test" to only process the last 48 chunks for a fast test run, else "no".
		total_parts: Total number of parts chunks are split into (for parallelization).
	"""
	import math
	import pandas as pd
	import glob
	import cellxgene_census
	import scanpy

	print("STEP 2 - Get census data chunks")

	## Read human_meta_counts_healthy-diseased.pkl if not already in memory
	human_meta_counts = pd.read_pickle(os.path.join(data_folder, "human_meta_counts_healthy-diseased.pkl"))

	## Get all unique gene names from LLM ontologies
	print("Retrieving gene names from LLM ontologies")
	tsv_files = glob.glob(os.path.join(llm_ontology_folder, "*_level2.tsv"))

	gene_names = set()
	for file in tsv_files:
		df = pd.read_csv(file, sep='\t', usecols=[0], header=None)
		gene_names.update(df.iloc[:, 0].dropna().unique())

	print(f"Total unique gene names: {len(gene_names)}")

	## Define filter to get data only for these genes
	var_value_filter="feature_id in [" + ", ".join(f'"{gene}"' for gene in gene_names) + "]"

	if test_mode == "test":
		print("Test mode: only process last 48 chunks")
		all_chunks = human_meta_counts.index[::-1][0:48] # Last 48 chunks for testing  
	else:
		all_chunks = human_meta_counts.index[:]
	num_chunks = len(all_chunks)
	chunks_per_part = math.ceil(num_chunks / total_parts)
	start_idx = part * chunks_per_part
	end_idx = min((part + 1) * chunks_per_part, num_chunks)
	selected_chunks = all_chunks[start_idx:end_idx]

	print("Processing part info")
	print(len(selected_chunks), part, start_idx, end_idx, chunks_per_part, num_chunks)

	for chunk_name in selected_chunks:	
		print(pd.Timestamp.now())
		print(f"Processing chunk: {chunk_name}")
		print(f"Number of cells in chunk: {human_meta_counts.loc[chunk_name[0],chunk_name[1],chunk_name[2],chunk_name[3]]}")
		if chunk_name[3] == "healthy":
			obs_value_filter = f"tissue_general in ['{chunk_name[0]}'] and cell_type_ontology_term_id in ['{chunk_name[1]}'] and sex in ['{chunk_name[2]}'] and disease in ['normal'] and is_primary_data == True"
		elif chunk_name[3] == "diseased":
			obs_value_filter = f"tissue_general in ['{chunk_name[0]}'] and cell_type_ontology_term_id in ['{chunk_name[1]}'] and sex in ['{chunk_name[2]}'] and disease not in ['normal'] and is_primary_data == True"

		with cellxgene_census.open_soma(census_version = "2025-11-08") as census:
			adata_chunk = cellxgene_census.get_anndata(
				census=census, 
				organism="Homo sapiens",
				var_value_filter=var_value_filter,
				obs_value_filter=obs_value_filter,  
			)
		census.close()

		print("Filter and downsample")
		scanpy.pp.filter_cells(adata_chunk, min_genes=int(perc_genes * len(gene_names)), inplace=True)
		if adata_chunk.shape[0] > max_chunk_size:
			adata_chunk = scanpy.pp.subsample(adata_chunk, n_obs=max_chunk_size, copy=True)
		
		print("Write chunk to h5ad file")
		scanpy.write(data_folder + f"{chunk_name[0].replace(' ', '-')}_{chunk_name[1].replace(':','-').replace(' ', '-')}_{chunk_name[2].replace(' ', '-')}_{chunk_name[3]}_chunk.h5ad", adata_chunk)

		del adata_chunk


def main():
	parser = argparse.ArgumentParser(
		description="Download and chunk CELLxGENE census data step by step."
	)
	parser.add_argument(
		"--step",
		type=str,
		default="all",
		choices=["1", "2", "all"],
		help="Which step to run: 1 (determine census chunks), 2 (get census data chunks), "
		"or 'all' to run every step in order.",
	)
	parser.add_argument(
		"--part",
		type=int,
		default=0,
		help="Index of the part to process in step 2 (for parallelization over chunks).",
	)
	parser.add_argument(
		"--test-mode",
		type=str,
		default="no",
		choices=["no", "test"],
		help="Use 'test' to only process the last 48 chunks in step 2 for a fast test run.",
	)
	parser.add_argument(
		"--total-parts",
		type=int,
		default=12,
		help="Total number of parts chunks are split into in step 2 (for parallelization).",
	)
	args = parser.parse_args()

	if args.step in ("1", "all"):
		step1()
	if args.step in ("2", "all"):
		step2(part=args.part, test_mode=args.test_mode, total_parts=args.total_parts)


if __name__ == "__main__":
	main()


