"""Pairwise-compare stripped ontology JSON files via OpenRouter LLM requests.

For every pairwise combination of ``*_ontology_stripped.json`` files in
``data/llm_ontologies/output_llms/`` (produced by ``strip_ontology_terms.py``),
send a comparison prompt to an OpenRouter model and store the resulting
answer as a structured JSON file.
"""
import base64
import glob
import itertools
import json
import os
import warnings

import requests

ontology_folder = "./data/llm_ontologies/output_llms"
comparison_folder = os.path.join(ontology_folder, "stripped_comparisons")

openrouter_api_key = os.environ.get("OPENROUTER_PREMIUM_API_KEY")
openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

# Model used to judge/compare the two ontologies
comparison_model = "openai/gpt-5.6-luna-pro"

# Placeholder prompt to be replaced later; must produce a structured JSON answer.
# Pure text, the two ontology files are attached separately, not interpolated here.
PROMPT_TEXT = """
Compare the provided json files containing each several biological dimensions. Your task is to match all dimensions across the two files based on their dimension name and short description based on a biological knowledge. Give a sorted, structured list of your matches from good to bad with an estimated score. Put at the end of the list not matchable terms with a score of zero.
Your answer should have strictly the following JSON schema and no other output. Include a top-level field for each model's name (as found in the source files), and a list of matches, sorted by descending match score. Each match should have a simple identifier, a score, and the list of dimension IDs from each model involved in that match (an empty list if there is no matching dimension on that side).
Make sure every dimension from both files appears at least once across the matches.
{
  "model1_name": "<name of first model/file>",
  "model2_name": "<name of second model/file>",
  "matches": [
    {
      "match_id": "match_01",
      "score": 0.9,
      "model1_dimensions": ["dim1_this_is_a_id"],
      "model2_dimensions": ["dim4_this_is_a_id", "dim10_this_is_another_id"]
    },
    {
      "match_id": "match_02",
      "score": 0.0,
      "model1_dimensions": ["dim16_this_is_non_matching_id"],
      "model2_dimensions": []
    }
  ]
}
"""


def build_file_attachment(filename: str, text_content: str) -> dict:
	"""Build an OpenRouter "file" content block for a text/JSON attachment.

	Mirrors how a chat UI attaches a file alongside a typed message: the file
	is sent as a base64 data URI next to the user's text content block.

	Args:
		filename: Name shown for the attached file.
		text_content: Raw text content of the file.

	Returns:
		An OpenRouter message content block of type "file".
	"""
	encoded = base64.b64encode(text_content.encode("utf-8")).decode("ascii")
	return {
		"type": "file",
		"file": {
			"filename": filename,
			"file_data": f"data:application/json;base64,{encoded}",
		},
	}


def get_openrouter_answer(
	question: str,
	model_name: str,
	attachments: list[dict] | None = None,
	reasoning_effort: str = "minimal",
	web_search: bool = True,
	max_search_results: int = 3,
) -> str:
	"""Get answer from OpenRouter.

	Args:
		question: The input question.
		model_name: The name of the model to use.
		attachments: Optional list of OpenRouter file content blocks (see
			``build_file_attachment``) sent alongside the text, like attaching
			files in a chat interface.
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
	content = [{"type": "text", "text": question}] + (attachments or [])
	payload = {
		"model": model_name,
		"messages": [
		{
			"role": "user",
			"content": content,
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
		warnings.warn(f"OpenRouter API request failed with status code {response.status_code}: {response.text}")
		return ""
	if "choices" not in response.json() or len(response.json()["choices"]) == 0:
		warnings.warn(f"OpenRouter API response missing 'choices': {response.json()}")
		return ""
	return response.json()["choices"][0]["message"]["content"]


def main() -> None:
	pattern = os.path.join(ontology_folder, "*_ontology_stripped.json")
	stripped_files = sorted(glob.glob(pattern))

	os.makedirs(comparison_folder, exist_ok=True)

	for filepath_a, filepath_b in itertools.combinations(stripped_files, 2):
		key_a = os.path.basename(filepath_a)[:-len("_ontology_stripped.json")]
		key_b = os.path.basename(filepath_b)[:-len("_ontology_stripped.json")]

		output_file = os.path.join(comparison_folder, f"{key_a}__vs__{key_b}.json")
		if os.path.exists(output_file):
			print(f"Output file {output_file} already exists. Skipping...")
			continue

		with open(filepath_a, "r", encoding="utf-8") as f:
			ontology_a_text = f.read()
		with open(filepath_b, "r", encoding="utf-8") as f:
			ontology_b_text = f.read()

		attachments = [
			build_file_attachment(os.path.basename(filepath_a), ontology_a_text),
			build_file_attachment(os.path.basename(filepath_b), ontology_b_text),
		]

		print(f"Comparing {key_a} vs {key_b}...")
		answer = get_openrouter_answer(question=PROMPT_TEXT, model_name=comparison_model, attachments=attachments)

		with open(output_file, "w", encoding="utf-8") as f:
			f.write(answer)  # expected to already be a JSON-formatted string


if __name__ == "__main__":
	main()
