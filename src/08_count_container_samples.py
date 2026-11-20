"""Report the number of samples per split (train, valid, test) in the acx containers.

Loops over the census acx container pickle files for the train, tune and holdout
splits in data/large_sc_data/ and reports the number of samples in each available
container split. Results are printed and written to a text file.

Containers are loaded strictly one at a time and released (del + gc) before the
next one is loaded, so peak RAM is bounded by the largest single container.

Example usage:
	python ./src/08_count_container_samples.py
"""

import gc
import os
import pickle
import time

# Get from environment variable if set, else use current working directory
root_folder = os.getenv("WORKSPACE_FOLDER") or "."

data_final_folder = os.path.join(root_folder, "data/large_sc_data/")
results_folder = os.path.join(root_folder, "results/")
output_file = os.path.join(results_folder, "container_sample_counts.txt")

CONTAINER_NAMES = ["train", "tune", "holdout"]
SPLITS = ["train", "valid", "test"]


def count_split_samples(acx_container):
	"""Return {split: n_samples or None} for an acx DatasetContainer."""
	counts = {}
	for split in SPLITS:
		dataset = getattr(acx_container, split, None)
		if dataset is None:
			counts[split] = None
		elif getattr(dataset, "data", None) is not None:
			counts[split] = int(dataset.data.shape[0])
		else:
			counts[split] = len(dataset.sample_ids)
	return counts


def load_and_count(container_file):
	"""Load a single container, count its samples and free the memory again."""
	with open(container_file, "rb") as f:
		acx_container = pickle.load(f)
	counts = count_split_samples(acx_container)
	del acx_container
	gc.collect()
	return counts


def format_table(results):
	"""Format {container: {split: n}} as a plain text table."""
	header = f"{'container':<10}" + "".join(f"{s:>14}" for s in SPLITS) + f"{'total':>14}"
	lines = [header, "-" * len(header)]
	for name, counts in results.items():
		if counts is None:
			lines.append(f"{name:<10}  file not found")
			continue
		cells = "".join(f"{('-' if counts[s] is None else f'{counts[s]:,}'):>14}" for s in SPLITS)
		total = sum(n for n in counts.values() if n is not None)
		lines.append(f"{name:<10}{cells}{total:>14,}")
	return "\n".join(lines)


def main():
	results = {}
	for name in CONTAINER_NAMES:
		container_file = os.path.join(data_final_folder, f"census-acxcontainer_{name}.pkl")
		if not os.path.isfile(container_file):
			print(f"[{name}] Not found: {container_file}", flush=True)
			results[name] = None
			continue

		size_gb = os.path.getsize(container_file) / 1024**3
		print(f"[{name}] Loading {container_file} ({size_gb:.2f} GB) ...", flush=True)
		start = time.time()
		results[name] = load_and_count(container_file)
		print(f"[{name}] {results[name]} ({time.time() - start:.1f} s)", flush=True)

	table = format_table(results)
	print("\nSamples per container split:\n" + table, flush=True)

	os.makedirs(results_folder, exist_ok=True)
	with open(output_file, "w") as f:
		f.write(table + "\n")
	print(f"\nWritten to {output_file}")


if __name__ == "__main__":
	main()
