"""Build LLM-derived cell ontologies for AUTOENCODIX.

Each step of the pipeline can be run individually via the ``--step``
command line argument, e.g.::

	python 00_llm_ontologies.py --step 1
	python 00_llm_ontologies.py --step 2
	python 00_llm_ontologies.py --step 3
	python 00_llm_ontologies.py --step 4
	python 00_llm_ontologies.py --step all 
"""
import argparse
import json
import os
import warnings


import mygene
import requests

# ---- Shared configuration ----
# number of latent dimensions
latent_dims = [16, 32, 48]
second_level_range = "5-20"
# Ontology sources to consider
ontology_sources = ["KEGG", "Reactome", "GO (Gene Ontology)"]
# Organism
organism = "human"

# Get from environment variable if set, else use current working directory
root_folder = os.getenv("WORKSPACE_FOLDER") or "."
print(f"Using root folder: {root_folder}")

# Example json format
example_json_file = os.path.join("./data/llm_ontologies/example.json")

prompt_folder = os.path.join(root_folder, "data/llm_ontologies/prompts")
output_folder = os.path.join(root_folder, "data/llm_ontologies/output_llms")
ontology_folder = os.path.join(root_folder, "data/llm_ontologies/output_llms")
final_folder = os.path.join(root_folder, "data/llm_ontologies/final_ontologies")

model_list_openrouter = {
	"KimiK3" : "moonshotai/kimi-k3",
	"ClaudeOpus5" : "anthropic/claude-opus-5",
	"GPT5.6_TerraPro": "openai/gpt-5.6-terra-pro",
}


openrouter_api_key = os.environ.get("OPENROUTER_PREMIUM_API_KEY")
openrouter_url = "https://openrouter.ai/api/v1/chat/completions"


def get_openrouter_answer(
	question: str,
	model_name: str,
	reasoning_effort: str = "minimal",
	web_search: bool = True,
	max_search_results: int = 3,
) -> str:
	"""Get answer from OpenRouter.

	Args:
		question: The input question.
		model_name: The name of the model to use.
		reasoning_effort: Reasoning effort level passed to the OpenRouter
			`reasoning` parameter. One of "max", "xhigh", "high", "medium",
			"low", "minimal" or "none". Set to None to omit reasoning control.
		web_search: Whether to enable the OpenRouter "web" plugin for
			grounding the answer with web search results.
		max_search_results: Maximum number of web search results to use
			(only applied when ``web_search`` is True).

	Returns:
		Generated response text.
	"""

	payload = {
		"model": model_name,
		"messages": [
		{
			"role": "user",
			"content": question,
		}
		],
	}
	if reasoning_effort is not None:
		payload["reasoning"] = {"effort": reasoning_effort}
	if web_search:
		payload["plugins"] = [{"id": "web", "max_results": max_search_results}]

	response = requests.post(
	url=openrouter_url,
	headers={
		"Authorization": f"Bearer {openrouter_api_key}",
	},
	data=json.dumps(payload)
	)
	if response.status_code != 200:
		# Warn
		warnings.warn(f"OpenRouter API request failed with status code {response.status_code}: {response.text}")
		# Return empty response
		return ""
	if "choices" not in response.json() or len(response.json()["choices"]) == 0:
		warnings.warn(f"OpenRouter API response missing 'choices': {response.json()}")
		return ""
	return response.json()["choices"][0]["message"]["content"]


def get_genes_for_id(identifier, scope):
    """
    Retrieves HUGO gene symbols for a given pathway or GO identifier.

    Args:
        identifier (str): The ID to query (e.g., 'GO:0007173').
        scope (str): The type of identifier (e.g., 'go', 'pathway.reactome').

    Returns:
        list: A list of gene symbols, or an empty list if none are found.
    """
    mg = mygene.MyGeneInfo()
    # We query for the identifier and ask for only the 'symbol' field back.
    # The 'hits' key contains the list of matching genes.
    if isinstance(identifier, str):
        result = mg.query(identifier, scopes=scope, fields=['symbol','ensembl.gene'], species='human', fetch_all=True, verbose=False)
 
    genes_symbols = []
    genes_ensembl = []
    for hit in result:
        if 'symbol' in hit:
            genes_symbols.append(hit['symbol'])
        if 'ensembl' in hit:
            ensembl_data = hit['ensembl']
            if isinstance(ensembl_data, list):
                for ens in ensembl_data:
                    if 'gene' in ens:
                        genes_ensembl.append(ens['gene'])
            elif isinstance(ensembl_data, dict) and 'gene' in ensembl_data:
                genes_ensembl.append(ensembl_data['gene'])
    return genes_symbols, genes_ensembl


def step1():
	"""Step 1: Define prompts for each latent dimension and save them to txt files."""
	# Read example json format
	with open(example_json_file, "r") as f:
		example_json = f.read()

	prompt_combinations = []

	for dims in latent_dims:	

		prompt_template = f"""Define a list of exactly {dims} dimensions which describe the state, functions, metabolism and role of cells in {organism}.
	The dimensions should be distinct and comprise all relevant aspects of cellular biology to discriminate all cells in the {organism} body along those dimensions.
	The dimensions should be useful to classify and compare cells within an organism like {organism}.
	Phrase and adjust the dimensions such that they measurable by gene expression data.
	Finally, assign each of the {dims} dimension the {second_level_range} most relevant and characterizing pathways or ontology terms with their respective identifier from databases like {" or ".join(ontology_sources)}.
	Please, avoid largely overlapping and duplicated terms from different sources. For example, use only TCA cycle from either KEGG or Reactome but not both as terms or pathways linked to a dimension.
	Try to identify the {second_level_range} most important but distinct subterms regardless of their database source. 
    Find a set of subterms and dimension which comprise most genes possible to cover a large part of the transcriptome.
    Give your answer in the following JSON format:
	{example_json}
	Make sure to follow the JSON format exactly without any additional text outside the JSON structure.
	"""
		prompt_combinations.append(prompt_template)

	### Save prompts to txt-file ###
	os.makedirs(prompt_folder, exist_ok=True)

	for i, prompt in enumerate(prompt_combinations):
		with open(os.path.join(prompt_folder, f"prompt_{latent_dims[i]}.txt"), "w") as f:
			f.write(prompt)


def step2():
	"""Step 2: Query LLM for ontologies via openrouter and save output.

	Args:
		max_tokens: Maximum number of tokens to generate per OpenRouter completion.
	"""
	# Iterate over prompt files (dimensions) and query LLMs
	os.makedirs(output_folder, exist_ok=True)

	for dim in latent_dims:
		prompt_file = os.path.join(prompt_folder, f"prompt_{dim}.txt")
		with open(prompt_file, "r") as f:
			prompt_text = f.read()
		
		for model_name, model_id in model_list_openrouter.items():
			print(f"Querying {model_name} for {dim} dimensions...")
			# Skip if output file already exists
			output_file = os.path.join(output_folder, f"Dim{dim}_{model_name}_ontology.json")
			if os.path.exists(output_file):
				print(f"Output file {output_file} already exists. Skipping...")
				continue
			answer = get_openrouter_answer(question=prompt_text, model_name=model_id)
			
			# Save the answer to a JSON file
			output_file = os.path.join(output_folder, f"Dim{dim}_{model_name}_ontology.json")
			with open(output_file, "w") as f:
				f.write(answer)


def step3():
	"""Step 3: Retrieve Gene IDs for Ontology Terms."""
	ontology_files = [f for f in os.listdir(ontology_folder) if f.endswith('ontology.json')] # exclude "_with_genes" files

	ontologies = []
	for filename in ontology_files:
		print(f"Processing ontology file: {filename}")
		filepath = os.path.join(ontology_folder, filename)
		with open(filepath, 'r', encoding='utf-8') as f:
			ontologies.append(json.load(f))

	# Mapping from ontology source to mygene scope
	source_to_scope = {
		'go': 'go',
		'reactome': 'pathway.reactome',
		'kegg': 'pathway.kegg'
	}
	i_ont = 0
	for ontology in ontologies:
		# Each ontology dict has a single key, e.g., 'gemini_ontology'
		ontology_name = list(ontology.keys())[0]
		print(f"Processing ontology: {ontology_name}")
		# Check if _with_genes.json file already exists
		output_file = os.path.join(ontology_folder, f"{ontology_name}_with_genes.json")
		if os.path.exists(output_file):
			print(f"Output file {output_file} already exists. Skipping gene retrieval for this ontology.")
			i_ont += 1
			continue
		dimensions = ontology[ontology_name].keys()
		print(ontology_name)
		for dim in dimensions:
			print(dim)
			t_int = 0
			for term in ontology[ontology_name][dim]['ontology_terms']:
				
				# Each term is a dict with 'id' and 'source' keys
				# We will add a 'genes' key to each term with the list of gene symbols
				term_id = term['id']
				source = term['source'].lower()
				scope = source_to_scope.get(source)
				if term_id and scope:
					# catch timeout and repeat query (maximal 3 times)
					try_count = 0
					while try_count < 3:
						try:
							genes_symbols, genes_ensembl = get_genes_for_id(term_id, scope)
							break  # Break the loop if successful
						except Exception as e:
							try_count += 1
							print(f"Error retrieving genes for {term_id} (try {try_count}/3): {e}")
							if try_count == 3:
								print(f"Failed to retrieve genes for {term_id} after 3 attempts.")
								genes_symbols, genes_ensembl = [], []
					term['genes_hugo'] = genes_symbols  # Add the list of gene symbols to the term
					term['ensembl'] = genes_ensembl  # Add the list of Ensembl gene IDs to the term

				ontologies[i_ont][ontology_name][dim]['ontology_terms'][t_int] = term  # Update the terms with genes
				t_int += 1
		# Save the updated ontology with genes to a new JSON file
		print(f"Saving updated ontology with genes to {ontology_name}_with_genes.json")
		output_file = os.path.join(ontology_folder, f"{ontology_name}_with_genes.json")
		with open(output_file, 'w', encoding='utf-8') as f:
			json.dump(ontologies[i_ont], f, ensure_ascii=False, indent=2)
		
		i_ont += 1

	return ontologies


def step4(ontologies=None):
	"""Step 4: Infos about ontologies and Saving."""
	if ontologies is None:
		# Load the ontologies enriched with genes (output of step 3) from disk
		ontology_files = [f for f in os.listdir(ontology_folder) if f.endswith('_with_genes.json')]
		ontologies = []
		for filename in ontology_files:
			filepath = os.path.join(ontology_folder, filename)
			with open(filepath, 'r', encoding='utf-8') as f:
				ontologies.append(json.load(f))

	# Calculate the number of unique genes for each ontology
	ontology_unique_gene_counts = {}
	ontology_gene_sets = {}

	for ontology_dict in ontologies:
		ontology_name = list(ontology_dict.keys())[0]
		all_genes = set()
		for dim in ontology_dict[ontology_name]:
			for term in ontology_dict[ontology_name][dim]['ontology_terms']:
				if 'genes_hugo' in term:
					all_genes.update(term['genes_hugo'])
		ontology_unique_gene_counts[ontology_name] = len(all_genes)
		ontology_gene_sets[ontology_name] = all_genes

	print(ontology_unique_gene_counts)

	# Calculate the number of overlapping genes between all ontology_name
	ontology_names = list(ontology_gene_sets.keys())
	if len(ontology_names) > 1:
		overlap_genes = set.intersection(*(ontology_gene_sets[name] for name in ontology_names))
		print(f"Number of overlapping genes among all ontologies: {len(overlap_genes)}")
		union_genes = set.union(*(ontology_gene_sets[name] for name in ontology_names))
		print(f"Number of genes in union of all ontologies: {len(union_genes)}")
	else:
		print("Only one ontology present, no overlap calculation.")

	os.makedirs(final_folder, exist_ok=True)
	## Save ontology in AUTOENCODIX format
	for ontology_dict in ontologies:
		ontology_name = list(ontology_dict.keys())[0]
		level1_rows = []
		level2_rows = []
		for dim in ontology_dict[ontology_name]:
			for term in ontology_dict[ontology_name][dim]['ontology_terms']:
				term_id = term.get('id', '')
				# Level 1: term_id <tab> dimension
				level1_rows.append(f"{term_id}\t{dim}")
				# Level 2: ensembl_gene <tab> term_id
				for ensembl_id in term.get('ensembl', []):
					level2_rows.append(f"{ensembl_id}\t{term_id}")

		# Write level1 file
		level1_filename = f"{ontology_name}__ensembl_level1.tsv"
		with open(os.path.join(final_folder, level1_filename), 'w', encoding='utf-8') as f1:
			f1.write('\n'.join(level1_rows))

		# Write level2 file
		level2_filename = f"{ontology_name}__ensembl_level2.tsv"
		with open(os.path.join(final_folder, level2_filename), 'w', encoding='utf-8') as f2:
			f2.write('\n'.join(level2_rows))


STEPS = {
	"1": step1,
	"2": step2,
	"3": step3,
	"4": step4,
}


def main():
	parser = argparse.ArgumentParser(
		description="Build LLM-derived cell ontologies for AUTOENCODIX step by step."
	)
	parser.add_argument(
		"--step",
		type=str,
		default="all",
		choices=["1", "2", "3", "4", "all"],
		help="Which step to run: 1 (define prompts), 2 (query LLMs), "
		"3 (retrieve gene IDs), 4 (summarize and save), or 'all' to run every step in order.",
	)

	args = parser.parse_args()

	if args.step == "all":
		step1()
		step2()
		ontologies = step3()
		step4(ontologies)
	elif args.step == "2":
		step2()
	else:
		STEPS[args.step]()


if __name__ == "__main__":
	main()
