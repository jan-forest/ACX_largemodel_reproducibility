"""Final training of large disentanglix models 
Training is performed for each dimensionality separately specified by the first command line argument.

Example usage:
	python ./src/04c_large_disentanglix_finaltraining.py Dim16_Disentanglix

"""

#### Step 0 - Definitions #####
import os
import sys
import pickle
import autoencodix as acx
from autoencodix.configs.disentanglix_config import DisentanglixConfig

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

results_folder_tuning = os.path.join(root_folder, "results/large_disentanglix_save/run_Aug26/")
results_folder_save = os.path.join(root_folder, "results/large_disentanglix_save/run_Aug26/")

# Create folder results_folder_save if it doesn't exist
os.makedirs(results_folder_save, exist_ok=True)

dim_from_cli = sys.argv[1]  # "Dim16_Varix", "Dim32_Varix", ...

file_processed = os.path.join(data_final_folder, "census-acxcontainer_train.pkl")

#### Step 2 - Load data and define config #####
with open(file_processed, "rb") as f:
	acx_container = pickle.load(f)


scconfig = DisentanglixConfig(
	## Fixed params
	epochs=500,	
	checkpoint_interval= 500,
	loss_reduction= "sum",
	## Tunable params -> from estimation from Ontix runs
	batch_size= 2048,
	drop_p = 0.1, 
	enc_factor = 2, 
	weight_decay= 1e-3, 
	beta_mi = 0.5*1e-4,
	beta_tc = 10,
	beta_dimKL= 0.5*1e-4,
	learning_rate= 0.001, 
	n_layers= 2,
	latent_dim= int(dim_from_cli.split('_')[0][3:]),
	save_vram=True,
	save_memory=True,
)

#### Step 4 - Final training #####
disentanglix = acx.Disentanglix(
	data=acx_container,
	config=scconfig
	)

print("Starting preprocessing ...")
disentanglix.preprocess()
print("Starting training ...")
disentanglix.fit()
print("Visualizing losses ...")
disentanglix.visualize()

print("Saving plots ...")
disentanglix.visualizer.save_plots(os.path.join(results_folder_save, f"large_disentanglix_final_model_{dim_from_cli}_plots/"))
print("Training finished, saving model ...")
disentanglix.save(os.path.join(results_folder_save, f"large_disentanglix_final_model_{dim_from_cli}.pkl"), save_all=False)