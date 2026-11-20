import os
import pickle
import sys

import anndata
import celltypist
import matplotlib.pyplot as plt
import mygene
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import autoencodix as acx

# ── Configuration ────────────────────────────────────────────────────────────
# Get from environment variable if set, else use current working directory
root_folder = os.getenv("WORKSPACE_FOLDER") or "."

RESULTS_FOLDER = os.path.join(root_folder, "./results/large_varix_save/run_Aug26/")
DATA_FINAL_FOLDER = os.path.join(root_folder, "./data/large_sc_data/")
LLM_ONTOLOGY_FOLDER = os.path.join(root_folder, "./data/llm_ontologies/final_ontologies/")


# Loop variables -> not loop, but parallel on HPC
ONT_FROM_CLI_LIST = [
    sys.argv[1]  # "Dim16_ClaudeOpus5", "Dim16_Varix", "Dim16_Disentanglix", ...
]

MODEL_TYPE_LIST = [
    "Healthy_Human_Liver.pkl",
    "Adult_Human_Skin.pkl",
    "Adult_Human_Vascular.pkl",
    "Human_Lung_Atlas.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
]

CELLS_TO_KEEP = {
    "Immune_All_High.pkl": [
        "B cell",
        "T cell",
        "natural killer cell",
        "macrophage",
        "monocyte",
    ],
    "Immune_All_Low.pkl": [
        "B cell",
        "T cell",
        "natural killer cell",
        "macrophage",
        "monocyte",
    ],
    "Healthy_Human_Liver.pkl": "ALL",
    "Adult_Human_Skin.pkl": "ALL",
    "Adult_Human_Vascular.pkl": "ALL",
    "Human_Lung_Atlas.pkl": "ALL",
}
TISSUE_TO_KEEP = {
    "Immune_All_High.pkl": "ALL",
    "Immune_All_Low.pkl": "ALL",
    "Healthy_Human_Liver.pkl": ["liver"],
    "Adult_Human_Skin.pkl": ["skin of body"],
    "Adult_Human_Vascular.pkl": "ALL",
    "Human_Lung_Atlas.pkl": ["lung"],
}

TOP_X = 10
DOWNSAMPLE_LIMIT = 100000  # reduce number of cells if exceeded
# Create output directory for confusion matrices
celltypist_comparison_folder = os.path.join(RESULTS_FOLDER, "celltypist_comparison")
os.makedirs(celltypist_comparison_folder, exist_ok=True)

# Initialize performance_df to store accuracy results
performance_df = pd.DataFrame(
    columns=["Ontology", "Model", "Tissue", "Accuracy", "Imbalance"]
)


# ── Helper functions ─────────────────────────────────────────────────────────


def keep_features_from_acxcontainer(acx_container, feature_ids_to_keep):
    for split in ["train", "valid", "test"]:
        if getattr(acx_container, split) is None:
            continue

        dataset = getattr(acx_container, split)
        feature_id_array = np.array(dataset.feature_ids)
        keep_indices = [
            i for i, fid in enumerate(feature_id_array) if fid in feature_ids_to_keep
        ]

        dataset.data = dataset.data[:, keep_indices]
        dataset.feature_ids = [dataset.feature_ids[i] for i in keep_indices]
        setattr(acx_container, split, dataset)

    return acx_container


# ── Load shared data ──────────────────────────────────────────────────────────

file_holdout = os.path.join(DATA_FINAL_FOLDER, "census-acxcontainer_holdout.pkl")
print("Loading holdout data ...")
with open(file_holdout, "rb") as f:
    acx_container = pickle.load(f)

mg = mygene.MyGeneInfo()

# ── Main loop ─────────────────────────────────────────────────────────────────

for ont_from_cli in ONT_FROM_CLI_LIST:
    print(f"\n{'='*60}")
    print(f"Processing ontology: {ont_from_cli}")
    print(f"{'='*60}")

    # Load trained model
    final_varix_model_file = os.path.join(
        RESULTS_FOLDER, f"large_varix_final_model_{ont_from_cli}.pkl"
    )
    print("Loading trained model ...")
    loaded_varix = acx.Varix.load(file_path=final_varix_model_file)
    loaded_varix._trainer._config.save_vram = True

    # Load latent representations
    latent_holdout_file = os.path.join(
        RESULTS_FOLDER, f"holdout_latent_{ont_from_cli}.parquet"
    )
    print("Loading holdout set latent representations ...")
    df_latent = pd.read_parquet(latent_holdout_file)

    # Map Ensembl IDs to gene symbols
    # retry when receives HTTPStatusError
    success = False
    try_count = 0
    while not success and try_count < 5:
        try:
            mygene_mapping = mg.querymany(
                acx_container.test.feature_ids,
                scopes="ensembl.gene",
                fields="symbol",
                species="human",
                as_dataframe=True,
                returnall=False,
            )
            success = True
        except Exception as e:
            try_count += 1
            print(f"Error occurred while querying mygene: {e}")
            print("Retrying...")
            
    mygene_mapping["Ensembl_ID"] = mygene_mapping.index
    # Remove duplicates in the mapping (keep first occurrence)
    mygene_mapping = mygene_mapping[~mygene_mapping.index.duplicated(keep="first")]
    mygene_mapping = mygene_mapping.set_index("symbol")

    # ── Inner loop over celltypist model types ────────────────────────────────

    for model_type in MODEL_TYPE_LIST:
        print(f"\n  -- CellTypist model: {model_type}")

        # Select cell types
        CELLS_TO_KEEP_FOR_MODEL = CELLS_TO_KEEP.get(model_type, "ALL")
        TISSUE_TO_KEEP_FOR_MODEL = TISSUE_TO_KEEP.get(model_type, "ALL")
        sel_samples = [
            i
            for i, ct in enumerate(acx_container.test.metadata.cell_type)
            if ct in CELLS_TO_KEEP_FOR_MODEL or CELLS_TO_KEEP_FOR_MODEL == "ALL"
        ]
        # Select by tissue as well, if specified
        if TISSUE_TO_KEEP_FOR_MODEL != "ALL":
            sel_samples = [
                i
                for i in sel_samples
                if acx_container.test.metadata.tissue_general.iloc[i]
                in TISSUE_TO_KEEP_FOR_MODEL
            ]
        # Downsample if too many samples
        if len(sel_samples) > DOWNSAMPLE_LIMIT:
            sel_samples = np.random.choice(
                sel_samples, DOWNSAMPLE_LIMIT, replace=False
            ).tolist()
            print(
                f"  Downsampled to {DOWNSAMPLE_LIMIT} samples for performance reasons."
            )
        input_expression = acx_container.test.data[sel_samples, :]
        sel_samples = acx_container.test.metadata.index[sel_samples].tolist()
        print(
            f"Selected {len(sel_samples)} samples from the holdout set for generation."
        )

        # Generate reconstructions
        latent_priors = df_latent.loc[sel_samples, :].to_numpy()
        generated_reconstructions = loaded_varix.generate(latent_prior=latent_priors)
        print("Generated reconstructions shape:", generated_reconstructions.shape)

        # Build AnnData objects
        recon_data = generated_reconstructions.detach().cpu().numpy()
        recon_data = np.clip(recon_data, 0, 9)

        input_data = input_expression.copy()
        input_data[input_data < 0] = 0
        input_data[input_data > 9] = 9

        adata_recon = anndata.AnnData(X=recon_data, var=mygene_mapping)
        adata_input = anndata.AnnData(X=input_data, var=mygene_mapping)

        predictions_recon = celltypist.annotate(
            adata_recon, model=model_type, majority_voting=True, mode="best match"
        )
        predictions_input = celltypist.annotate(
            adata_input, model=model_type, majority_voting=True, mode="best match"
        )

        input_labels = predictions_input.predicted_labels.predicted_labels.astype(str)
        reconstructed_labels = (
            predictions_recon.predicted_labels.predicted_labels.astype(str)
        )

        comparison_df = pd.DataFrame(
            {
                "Sample": sel_samples,
                "Input_Cell_Type": input_labels.values,
                "Reconstructed_Cell_Type": reconstructed_labels.values,
            }
        )

        num_matches = (input_labels.values == reconstructed_labels.values).sum()
        print(
            f"  Matches: {num_matches} / {len(comparison_df)} samples "
            f"({100 * num_matches / len(comparison_df):.1f}%)"
        )

        # Confusion matrix
        top_input_cell_types = input_labels.value_counts().nlargest(TOP_X).index
        mask = input_labels.isin(top_input_cell_types)
        filtered_input = input_labels[mask]
        filtered_recon = reconstructed_labels[mask]

        cm = confusion_matrix(
            filtered_input, filtered_recon, labels=top_input_cell_types
        )
        # Calculate accuracy
        accuracy = np.trace(cm) / np.sum(cm)

        # Calculate imbalance metric (normalized difference between upper and lower off-diagonal sums)
        off_diag_lower_sum = np.sum(np.tril(cm, k=-1))
        off_diag_upper_sum = np.sum(np.triu(cm, k=1))
        imbalance = (
            (off_diag_upper_sum - off_diag_lower_sum)
            / (off_diag_upper_sum + off_diag_lower_sum)
            if (off_diag_upper_sum + off_diag_lower_sum) > 0
            else 0
        )

        # Save in performance_df
        performance_df = pd.concat(
            [
                performance_df,
                pd.DataFrame(
                    {
                        "Ontology": [ont_from_cli],
                        "Model": [model_type],
                        "Tissue": [
                            (
                                TISSUE_TO_KEEP_FOR_MODEL
                                if TISSUE_TO_KEEP_FOR_MODEL != "ALL"
                                else "All"
                            )
                        ],
                        "Accuracy": [accuracy],
                        "Imbalance": [imbalance],
                    }
                ),
            ],
            ignore_index=True,
        )

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=top_input_cell_types,
            yticklabels=top_input_cell_types,
            cmap="Blues",
        )
        plt.xlabel("Reconstructed Cell Type")
        plt.ylabel("Input Cell Type")
        plt.title(
            f"Input vs Reconstructed Cell Types (Top {TOP_X})\n"
            f"{ont_from_cli} | {model_type}"
        )

        out_path = os.path.join(
            RESULTS_FOLDER,
            "celltypist_comparison",
            f"celltypist_{model_type.strip('.pkl')}_{ont_from_cli}.png",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved confusion matrix to {out_path}")

    # Save performance summary
    performance_summary_file = os.path.join(
        RESULTS_FOLDER, "celltypist_comparison", f"celltypist_performance_summary_{ont_from_cli}.csv"
    )
    performance_df.to_csv(performance_summary_file, index=False)
    print(f"\nSaved performance summary to {performance_summary_file}")
    print("\nDone.")
