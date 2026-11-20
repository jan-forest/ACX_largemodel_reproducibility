"""Prediction, visualization and evaluation of final large models on the holdout set.

The model architecture (Ontix, Varix or Disentanglix) is detected automatically from
the saved model file name that matches the identifier given as the first command line
argument. The identifier is the same one used when calling the corresponding
finaltraining script, e.g. "Dim16_ClaudeOpus5" (Ontix), "Dim16_Varix" (Varix) or
"Dim16_Disentanglix" (Disentanglix).

Example usage:
	python ./src/05_pred_vis_eval.py Dim16_ClaudeOpus5
	python ./src/05_pred_vis_eval.py Dim16_Varix
	python ./src/05_pred_vis_eval.py Dim16_Disentanglix

"""

#### Step 0 - Definitions #####
import os
import sys
import json
import pickle
import pandas as pd
import autoencodix as acx


def keep_features_from_acxcontainer(acx_container, feature_ids_to_keep):
	import numpy as np

	for split in ['train', 'valid', 'test']:
		if getattr(acx_container, split) is None:
			continue

		dataset = getattr(acx_container, split)
		feature_id_array = np.array(dataset.feature_ids)
		keep_indices = [i for i, fid in enumerate(feature_id_array) if fid in feature_ids_to_keep]

		dataset.data = dataset.data[:, keep_indices]
		dataset.feature_ids = [dataset.feature_ids[i] for i in keep_indices]
		setattr(acx_container, split, dataset)

	return acx_container


# Get from environment variable if set, else use current working directory
root_folder = os.getenv("WORKSPACE_FOLDER") or "."

data_final_folder = os.path.join(root_folder, "data/large_sc_data/")
llm_ontology_folder = os.path.join(root_folder, "data/llm_ontologies/final_ontologies/")
tasks_file_path = os.path.join("./data/gemini_celltype_tasks2.json")

run_folder_name = "run_Aug26"

# Per-architecture settings, keyed by model type name used in file/folder naming
MODEL_SPECS = {
	"ontix": {
		"class": acx.Ontix,
		"results_folder": os.path.join(root_folder, f"results/large_ontix_save/{run_folder_name}/"),
		"uses_ontology": True,
		"rf_max_depth": 8,
	},
	"varix": {
		"class": acx.Varix,
		"results_folder": os.path.join(root_folder, f"results/large_varix_save/{run_folder_name}/"),
		"uses_ontology": False,
		"rf_max_depth": 5,
	},
	"disentanglix": {
		"class": acx.Disentanglix,
		"results_folder": os.path.join(root_folder, f"results/large_disentanglix_save/{run_folder_name}/"),
		"uses_ontology": False,
		"rf_max_depth": 5,
	},
}

model_id_from_cli = sys.argv[1]  # "Dim16_ClaudeOpus5", "Dim16_Varix", "Dim16_Disentanglix", ...

file_holdout = os.path.join(data_final_folder, "census-acxcontainer_holdout.pkl")
file_tuning = os.path.join(data_final_folder, "census-acxcontainer_tune.pkl")

#### Step 0b - Detect model architecture from the saved model file name #####
model_type = None
final_model_file = None
for candidate_type, candidate_spec in MODEL_SPECS.items():
	candidate_file = os.path.join(candidate_spec["results_folder"], f"large_{candidate_type}_final_model_{model_id_from_cli}.pkl")
	print(f"Checking for model file: {candidate_file}")
	if os.path.exists(candidate_file):
		print(f"Found model file for architecture '{candidate_type}': {candidate_file}")
		model_type = candidate_type
		final_model_file = candidate_file
		break

if model_type is None:
	searched = [os.path.join(s["results_folder"], f"large_{t}_final_model_{model_id_from_cli}.pkl") for t, s in MODEL_SPECS.items()]
	raise FileNotFoundError(
		f"Could not find a trained model for identifier '{model_id_from_cli}'. Searched:\n" + "\n".join(searched)
	)

print(f"Detected model architecture: {model_type} (model file: {final_model_file})")
spec = MODEL_SPECS[model_type]
results_folder = spec["results_folder"]
model_class = spec["class"]

#### Step 1 - Load and predict on holdout set #####
ont_lvl2 = None
if spec["uses_ontology"]:
	print("Preparing holdout data for feature size per ontology ...")
	ont_files = [
		# Order from Latent Dim -> Hidden Dim -> Input Dim
		os.path.join(llm_ontology_folder, f"{model_id_from_cli}_ontology__ensembl_level1.tsv"),
		os.path.join(llm_ontology_folder, f"{model_id_from_cli}_ontology__ensembl_level2.tsv"),
		]
	ont_lvl2 = pd.read_csv(ont_files[1], sep='\t', usecols=[0], header=None)
	ont_lvl2.columns = ['feature_id']

print("Loading tune set to pre-fit PCA reducer ...")
with open(file_tuning, "rb") as f:
	acx_container = pickle.load(f)  # Later overwritten to save RAM

if ont_lvl2 is not None:
	acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)

## Pre-fit PCA reducer
from sklearn.decomposition import PCA

dim = int(model_id_from_cli.split("_")[0].replace("Dim", ""))

pca = PCA(n_components=dim)
pca.fit(acx_container.train._to_df())

# Save the pre-fitted PCA reducer as pickle
pca_reducer_file = os.path.join(results_folder, f"{model_id_from_cli}_pca_reducer.pkl")
with open(pca_reducer_file, "wb") as f:
	pickle.dump(pca, f)

print("Loading holdout data ...")
with open(file_holdout, "rb") as f:
	acx_container = pickle.load(f)

if ont_lvl2 is not None:
	acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)


#####  Expand metadata with cell type tasks ###################
with open(tasks_file_path, "r") as f:
	gemini_celltype_tasks2 = json.load(f)


print("Loading trained model ...")
loaded_model = model_class.load(file_path=final_model_file)
loaded_model._trainer._config.save_vram = True  # Enable memory saving for prediction on holdout set

print("Predicting on holdout data ...")
result = loaded_model.predict(
	data=acx_container
)

#### Step 2 - create plots ####

# UMAP representations of latent space
params_umap = ["sex", "is_diseased", "assay", "tissue_general", "high_level_stage_name"]

loaded_model.visualizer.show_latent_space(
	result=loaded_model.result,
	plot_type='2D-scatter',
	param=params_umap,
	split='test',
	n_downsample=20000)

# Ridgeline plots of latent space
params_ridge = params_umap + list(gemini_celltype_tasks2.keys())[0:5]  # Use the cell type tasks defined in the JSON file

loaded_model.visualizer.show_latent_space(
	result=loaded_model.result,
	plot_type='Ridgeline',
	param=params_ridge,
	split='test',
	n_downsample=20000)

# Heatmap representations of latent space
params_heatmap = params_umap

loaded_model.visualizer.show_latent_space(
	result=loaded_model.result,
	plot_type='Clustermap',
	param=params_heatmap,
	split='test',
	n_downsample=20000)


#### Step 3 - Evaluate embeddings ####
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

tasks = list(gemini_celltype_tasks2.keys()) + ["tissue_general", "sex", "disease", "high_level_stage_name"]
sklearn.set_config(enable_metadata_routing=True)

sklearn_ml_class = linear_model.LogisticRegression(
							solver="saga",
							n_jobs=-1,
							class_weight="balanced",
							max_iter=500,
)

sklearn_ml_regression = linear_model.LinearRegression()  ## Unused, only classification tasks
own_metric_class = 'roc_auc_ovo'
own_metric_regression = 'r2'

loaded_model.evaluate(
		ml_model_class=sklearn_ml_class,
		ml_model_regression=sklearn_ml_regression,
		params=tasks,
		metric_class=own_metric_class,
		metric_regression=own_metric_regression,
		reference_methods=["PCA"],  # No reference methods for tuning
		reference_reducer={"PCA": pca},  # Use the pre-fitted PCA reducer for the reference method
		split_type="CV-3",
		top_k_classes=20,
		n_downsample=None,  # Use a subset of the data for faster evaluation
		exclude_classes=["other", "unknown"],
	)

# Test RandomForest as additional model for evaluation
sklearn_ml_class = RandomForestClassifier(
							n_estimators=100,
							n_jobs=-1,
							max_depth=spec["rf_max_depth"],
							min_samples_leaf=4,
							class_weight="balanced",
)

loaded_model.evaluate(
		ml_model_class=sklearn_ml_class,
		ml_model_regression=sklearn_ml_regression,
		params=tasks,
		metric_class=own_metric_class,
		metric_regression=own_metric_regression,
		reference_methods=["PCA"],  # No reference methods for tuning
		reference_reducer={"PCA": pca},  # Use the pre-fitted PCA reducer for the reference method
		split_type="CV-3",
		top_k_classes=20,
		n_downsample=None,  # Use a subset of the data for faster evaluation
		exclude_classes=["other", "unknown"],
)

#### Step 4 - Save ####
# Save latent embeddings as dataframe
loaded_model.result.get_latent_df(epoch=-1, split='test').to_parquet(os.path.join(results_folder, f"holdout_latent_{model_id_from_cli}.parquet"))

# Save plots
path_plots = os.path.join(results_folder, f"large_{model_type}_holdout_{model_id_from_cli}_plots/")
os.makedirs(path_plots, exist_ok=True)
loaded_model.visualizer.save_plots(
	path=path_plots, which='all', format='png'
)

# Save evaluation results
path_eval = os.path.join(results_folder, f"large_{model_type}_holdout_{model_id_from_cli}_evaluation.csv")
loaded_model.result.embedding_evaluation.to_csv(path_eval, index=False)