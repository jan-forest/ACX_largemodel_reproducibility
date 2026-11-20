"""Tuning of large ontix models with syne_tune.
Tuning is performed for each ontology separately specified by the first command line argument.
The second command line argument specifies the maximum wallclock time in hours for tuning, and the third command line argument specifies the number of parallel workers for tuning.
Example usage:
	python ./src/03_large_ontix_tuning.py Dim16_ClaudeOpus5 24 4

"""

#### Step 0 - Definitions #####
import sys
import os
from datetime import datetime

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import PythonBackend
from syne_tune.config_space import randint, uniform, loguniform
from syne_tune.optimizer.baselines import CQR
from syne_tune.experiments import load_experiment


# Get from environment variable if set, else use current working directory
root_folder = os.getenv("WORKSPACE_FOLDER") or "."

data_final_folder = os.path.join(root_folder, "data/large_sc_data/")


results_folder = os.path.join(root_folder, "results/large_ontix_save/run_Aug26/")

# Create results_folder if it doesn't exist
if not os.path.exists(results_folder):
	os.makedirs(results_folder)

ont_from_cli = sys.argv[1]  # "Dim16_ClaudeOpus5", "Dim32_KimiK3", ...
max_wallclock_time = int(float(sys.argv[2])*60*60)  # given in hours translated to seconds
n_workers = int(sys.argv[3])  # number of parallel workers for tuning

metric = "ml_performance" # "ml_performance" or "recon_loss"

tasks = ["tissue_general", "sex", "disease", "high_level_stage_name"]

import json
tasks_file_path = os.path.join("./data/gemini_celltype_tasks2.json")

with open(tasks_file_path, "r") as f:
	gemini_celltype_tasks2 = json.load(f)
 
for task_name, cell_types in gemini_celltype_tasks2.items():
	if task_name not in tasks:
		tasks.append(task_name)

file_processed = os.path.join(data_final_folder, "census-acxcontainer_tune.pkl")


#### Step 1 - Definition of syne_trainer function #####
def syne_trainer(
	## Fixed params
	epochs: int,
	checkpoint_interval: int,
	loss_reduction: str,
	root_folder: str,
	data_path: str,
	ontology_name: str,
	tasks: str,
	## Tunable params
	batch_size: int,
	drop_p: float,
	enc_factor: int,
	weight_decay: float,
	beta: float,
	learning_rate: float,
	n_layers: int,
):
	## Load Dataset Container from pickle with already preprocessed data
	import os
	import pickle
	import pandas as pd
	import sklearn
	from sklearn import linear_model
	import autoencodix as acx
	from autoencodix.configs.ontix_config import OntixConfig
	from syne_tune import Reporter

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

	llm_ontology_folder = os.path.join(root_folder, "data/llm_ontologies/final_ontologies/")
	
	file_pkl = data_path

	with open(file_pkl, "rb") as f:
		acx_container = pickle.load(f)

	scconfig = OntixConfig(
		## Fixed params
		epochs=epochs,
		checkpoint_interval= checkpoint_interval,
		loss_reduction= loss_reduction,
		## Tunable params
		batch_size= batch_size,
		drop_p= drop_p,
		enc_factor= enc_factor,
		weight_decay= weight_decay,
		beta= beta,
		learning_rate= learning_rate,
		n_layers= n_layers,
		save_memory=False,
		save_vram=True,  # Enable VRAM saving
	)


	ont_files = [
		# Order from Latent Dim -> Hidden Dim -> Input Dim
		os.path.join(llm_ontology_folder, f"{ontology_name}_ontology__ensembl_level1.tsv"),
		os.path.join(llm_ontology_folder, f"{ontology_name}_ontology__ensembl_level2.tsv"),
		]

	ont_lvl2 = pd.read_csv(ont_files[1], sep='\t', usecols=[0], header=None)
	ont_lvl2.columns = ['feature_id']

	acx_container = keep_features_from_acxcontainer(acx_container, ont_lvl2['feature_id'].values)

	ontix = acx.Ontix(
		data=acx_container,
		ontologies=ont_files,
		config=scconfig
		)

	ontix.run()


	## Embedding Evaluation
	sklearn.set_config(enable_metadata_routing=True)

	sklearn_ml_class = linear_model.LogisticRegression(
								solver="sag",
								n_jobs=-1,
								class_weight="balanced",
								max_iter=200,
	) 
	sklearn_ml_regression = linear_model.LinearRegression() ## Unused, only classification tasks
	own_metric_class = 'roc_auc_ovo'  
	own_metric_regression = 'r2' 

	tasks_list = tasks.split("$")
	ontix.evaluate(
		ml_model_class=sklearn_ml_class, 
		ml_model_regression=sklearn_ml_regression, 
		params= tasks_list,	
		metric_class = own_metric_class, 
		metric_regression = own_metric_regression, 
		reference_methods = [], # No reference methods for tuning
		split_type = "use-split",
		top_k_classes = 20,
		n_downsample = None, # Use a subset of the data for faster evaluation
		exclude_classes = ["other"],
	)

	## Average of cell type task starting with "Task*"
	avg_celltype_performance = ontix.result.embedding_evaluation.loc[
		(ontix.result.embedding_evaluation.score_split == "valid") &
		(ontix.result.embedding_evaluation.CLINIC_PARAM.str.startswith("Task")),
		"value"
	].mean()
	## Average of all other tasks
	avg_other_performance = ontix.result.embedding_evaluation.loc[
		(ontix.result.embedding_evaluation.score_split == "valid") &
		(~ontix.result.embedding_evaluation.CLINIC_PARAM.str.startswith("Task")),
		"value"
	].mean()
	
	avg_mltask_performance = (avg_celltype_performance + avg_other_performance) / 2
	valid_recon_loss = float(ontix.result.sub_losses.get("recon_loss").get(epoch=-1, split="valid"))
 
	# Variance across tasks
	var_celltype_performance = ontix.result.embedding_evaluation.loc[
		(ontix.result.embedding_evaluation.score_split == "valid") &
		(ontix.result.embedding_evaluation.CLINIC_PARAM.str.startswith("Task")),
		"value"
	].var()
	var_other_performance = ontix.result.embedding_evaluation.loc[
		(ontix.result.embedding_evaluation.score_split == "valid") &
		(~ontix.result.embedding_evaluation.CLINIC_PARAM.str.startswith("Task")),
		"value"
	].var()

	report = Reporter()
	report(	ml_performance=avg_mltask_performance,
        	recon_loss=valid_recon_loss,
        	ml_perf_celltype=avg_celltype_performance,
			ml_perf_other=avg_other_performance,
			var_celltype_performance=var_celltype_performance,
			var_other_performance=var_other_performance,
			)
### Step 2 - Tuning #####


# Hyperparameter configuration 
epoch = 250
config_space = {
	## Fixed params
	"epochs": epoch,
	"checkpoint_interval": epoch,
	"loss_reduction": "sum",
	"root_folder": root_folder,
	"data_path": file_processed,
	"ontology_name": ont_from_cli,
	"tasks": "$".join(tasks),
	## Tunable params
	"batch_size": randint(64, 4096),
	"drop_p": uniform(0.0, 0.9),
	"enc_factor": uniform(1, 3),
	"weight_decay": loguniform(1e-5, 1e-1),
	"beta": loguniform(1e-5, 10),
	"learning_rate": loguniform(1e-5, 1e-1),
	"n_layers": randint(5, 5),
}

# Define whether to minimize or maximize the metric
if metric == "ml_performance":
	do_minimize = False
elif metric == "recon_loss":
	do_minimize = True
else:
	do_minimize = False

# Scheduler (i.e., HPO algorithm)
scheduler = CQR(
    config_space,
    metric=metric,
    do_minimize=do_minimize
)

# Tuning
print("Starting tuning ...")
tuner = Tuner(
    trial_backend=PythonBackend(tune_function=syne_trainer, config_space=config_space),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(
		max_wallclock_time=max_wallclock_time,  # in seconds
		print_update_interval = int(max_wallclock_time/120), # Adjust print update interval based on max wallclock time
		),
    n_workers=n_workers,  # how many trials are evaluated in parallel (GPUs)
)

tuner.run()
tuning_experiment = load_experiment(tuner.name)
print(f"best result found: {tuning_experiment.best_config()}")

#### Step 3 - Save tuning experiment results #####
import pickle

# current date for filename
current_date = datetime.now().strftime("%Y-%m-%d")

with open(os.path.join(results_folder, f"large_ontix_{ont_from_cli}_tuning.pkl"), "wb") as f:
	pickle.dump(tuning_experiment, f)