"""Create stripped copies of the LLM ontology JSON files.

For each ``*_ontology.json`` file in ``data/llm_ontologies/output_llms/``,
write a copy in the same folder where every dimension only keeps its
``dimension_name`` and the ``ontology_terms`` entry is removed.
"""
import glob
import json
import os

ontology_folder = "./data/llm_ontologies/output_llms"


def strip_ontology_terms(ontology: dict) -> dict:
	"""Remove the ``ontology_terms`` field from every dimension.

	Args:
		ontology: Parsed ontology JSON with a single top-level key mapping
			to a dict of dimensions.

	Returns:
		A new ontology dict with ``ontology_terms`` removed from each dimension.
	"""
	stripped = {}
	for ontology_name, dims in ontology.items():
		stripped[ontology_name] = {
			dim_name: {k: v for k, v in dim.items() if k != "ontology_terms"}
			for dim_name, dim in dims.items()
		}
	return stripped


def main() -> None:
	pattern = os.path.join(ontology_folder, "*_ontology.json")
	for filepath in glob.glob(pattern):
		filename = os.path.basename(filepath)
		with open(filepath, "r", encoding="utf-8") as f:
			ontology = json.load(f)

		stripped = strip_ontology_terms(ontology)

		stripped_filename = filename.replace("_ontology.json", "_ontology_stripped.json")
		stripped_filepath = os.path.join(ontology_folder, stripped_filename)
		with open(stripped_filepath, "w", encoding="utf-8") as f:
			json.dump(stripped, f, indent=2, ensure_ascii=False)
		print(f"Wrote {stripped_filepath}")


if __name__ == "__main__":
	main()
