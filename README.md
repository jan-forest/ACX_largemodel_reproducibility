# ACX_largemodel_reproducibility

Code, data references and plotting notebooks for the large pretrained autoencoder models presented in the AUTOENCODIX
publication *"AUTOENCODIX: powering a hub for biological representation learning models"* (citation will be added
when the preprint is available).

The repository documents how the training corpus was built from the CELLxGENE Census and how the models were
generated with LLM-derived ontologies, tuned, trained and evaluated. It also shows how every manuscript figure was
produced. All experiments were run with the Python package [AUTOENCODIX](https://github.com/jan-forest/autoencodix_package)
(`autoencodix==0.2.4`) on an HPC cluster with NVIDIA H100 GPUs.

> **Note for reviewers.** Most steps need an HPC cluster and run for hours to days, so a full re-run is not the
> intended use of this repository. All intermediate and final results are published (see [Data and results](#data-and-results)).
> The scripts and notebooks show exactly how each result and figure was produced, and you can re-run individual
> parts if needed. [How the manuscript figures were produced](#how-the-manuscript-figures-were-produced) maps each
> figure panel to its script and notebook.

## Contents

- [Data and results](#data-and-results)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [How the experiments were run](#how-the-experiments-were-run)
- [How the manuscript figures were produced](#how-the-manuscript-figures-were-produced)
- [License](#license)

## Data and results

| What | Where | Local path expected by the code |
|---|---|---|
| Training corpus (train / tune / holdout split as `.h5ad` and AUTOENCODIX container `.pkl`) | Hugging Face dataset [autoencodix/census-large](https://huggingface.co/datasets/autoencodix/census-large) | `data/large_sc_data/` |
| Trained models (15 models: 9 Ontix, 3 Varix, 3 Disentanglix) | Hugging Face collection [ACX pretrained models](https://huggingface.co/collections/autoencodix/acx-pretrained-models) | `results/large_<architecture>_save/run_Aug26/` |
| LLM-derived ontologies: prompts, raw LLM answers, gene-annotated JSON, pairwise dimension comparisons and final ontologies (`*_ontology__ensembl_level{1,2}.tsv`, also in the `ontology/` folder of each Ontix model repository on Hugging Face) | this repository | `data/llm_ontologies/` |
| Extended results (plots, evaluation tables, holdout latent embeddings, PCA baselines, tuning results) | Zenodo: [10.5281/zenodo.24396083](https://doi.org/10.5281/zenodo.24396083) | `results/` (unzip in the repository root) |
| Cell type classification tasks (Supplementary Table 1) | this repository | `data/gemini_celltype_tasks2.json` |
| Developmental stage ontology used for `high_level_stage_name` | this repository | `data/hsapdv.obo` |

Example download of the data and one model with the Hugging Face CLI (`hf`, part of `huggingface_hub`):

```bash
# Corpus (the train container alone is ~80 GB; select single files with --include)
hf download autoencodix/census-large --repo-type dataset --local-dir data/large_sc_data --include "census-acxcontainer_holdout.pkl"
# Model (the matching ontology is already in data/llm_ontologies/final_ontologies/)
hf download autoencodix/Ontix-Dim32-ClaudeOpus5 --local-dir results/large_ontix_save/run_Aug26 --exclude "ontology/*"
# Extended results from Zenodo
wget https://zenodo.org/records/24396083/files/ACX_largemodel_results.zip
unzip ACX_largemodel_results.zip   # in the repository root, restores results/
```

## Repository structure

```
├── run_0X_inHPC.sh        # SLURM submission scripts, one per pipeline stage (see below)
├── src/                   # pipeline steps called by the run scripts
├── notebooks/             # notebooks producing the manuscript figures from results/
├── data/                  # input data and LLM-derived ontologies (corpus is downloaded from Hugging Face, see above)
├── results/               # outputs of the pipeline (models on Hugging Face, the rest on Zenodo)
├── pyproject.toml
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.10.6.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install the dependencies (versions pinned in pyproject.toml)
uv venv --python 3.10.6
source .venv/bin/activate
uv sync
```

Environment variables:

```bash
# Root folder containing data/ and results/ (default: current directory).
# On the HPC this pointed to the project workspace.
export WORKSPACE_FOLDER="/path/to/workspace"

# OpenRouter API key, needed only for LLM queries (step 00, ontology comparison, notebook 06)
export OPENROUTER_PREMIUM_API_KEY="sk-or-..."
```

## How the experiments were run

All stages were submitted to a SLURM cluster with the `run_0X_inHPC.sh` scripts, in numerical order and from the
repository root. Each script contains the cluster-specific settings (`module load`, `#SBATCH` resources and account)
and submits one or more jobs that call the Python scripts in `src/`. To run on a different system, adapt the
`#SBATCH` header or call the Python commands inside the scripts directly.

**The number of a run script does not always match the number of the `src/` step it runs.** The table lists
which steps each script calls.

| Run script | Calls | What it does | Resources (per job) | Main outputs |
|---|---|---|---|---|
| `run_00_inHPC.sh` | `src/00_llm_ontologies.py --step 1…4` | Creates the prompts (Supp. Fig. 3), queries Claude Opus 5, GPT-5.6 TerraPro and Kimi K3 via OpenRouter for 16/32/48 dimensions, maps ontology terms to genes with MyGene and writes the final two-level ontologies | 1 job, < 2 h | `data/llm_ontologies/{prompts,output_llms,final_ontologies}/` (in this repository) |
| *(manual, optional)* | `src/strip_ontology_terms.py`, then `src/compare_stripped_ontologies.py` | Strips the LLM ontologies to dimension names and lets an LLM match dimensions between all pairs of ontologies (used in the extended analysis of `notebooks/00_llm_ontology_analysis.ipynb`) | local, minutes | `data/llm_ontologies/output_llms/stripped_comparisons/` (in this repository) |
| `run_01_inHPC.sh` | `src/01_get_census_data.py --step 1` | Queries the Census metadata (LTS 2025-11-08) and defines the chunks (cell type × tissue × sex × disease state) | 1 job, < 1 h | chunk definitions in `data/census_chunks/` |
| `run_02_inHPC.sh` | `src/01_get_census_data.py --step 2` | Retrieves the chunks from the Census, split into 64 parts processed as parallel jobs (downsampling to 1,000 cells per chunk). A commented fallback in the script reruns only selected parts. Set `test="test"` for a small test run. | ~64 jobs, up to 48 h | `data/census_chunks/*.h5ad` |
| `run_03_inHPC.sh` | `src/02_census_preparation.py --step 1`, `--step 2 --split {train,tune,holdout}` | Joins the chunks, adds annotations and classification tasks, splits into train (90%) / tune (5%) / holdout (5%), and builds AUTOENCODIX containers | 1 job, < 3 h, high memory | `data/large_sc_data/` (= Hugging Face dataset) |
| `run_04_inHPC.sh` | `src/03_large_ontix_tuning.py <ontology> 48 4` | Hyperparameter optimisation with Syne Tune for each of the 9 Ontix models on the tune split | 9 jobs, 48 h each, 3 nodes × 4 H100 | `results/large_ontix_save/run_Aug26/large_ontix_<model>_tuning.pkl` |
| `run_05_inHPC.sh` | `src/04_large_ontix_finaltraining.py`, `src/04b_large_varix_finaltraining.py`, `src/04c_large_disentanglix_finaltraining.py` | Final training of the 9 Ontix models (best tuning configuration, 500 epochs) and of the Varix and Disentanglix baselines (16/32/48 dimensions) on the train split | 15 jobs, up to 96 h, 1 H100 | `results/large_*_save/run_Aug26/large_*_final_model_<model>*` (= Hugging Face models), training loss plots |
| `run_06_inHPC.sh` | `src/05_pred_vis_eval.py <model>` | Prediction on the holdout split, latent space plots, PCA baseline and downstream task evaluation for all 15 models | 15 jobs, < 2 h, 1 GPU | `holdout_latent_<model>.parquet`, `<model>_pca_reducer.pkl`, `large_*_holdout_<model>_evaluation.csv`, `large_*_holdout_<model>_plots/` |
| `run_07_inHPC.sh` | `src/07_generate_with_ontix.py`, `src/07b_generate_with_varix.py` | Data generation from holdout embeddings and comparison of CellTypist annotations before and after reconstruction (Ontix and Varix models) | 12 jobs, < 1 h, 1 GPU | `results/large_{ontix,varix}_save/run_Aug26/celltypist_comparison/` |
| `run_08_inHPC.sh` | `src/08_count_container_samples.py` | Counts the cells per split in each container | 1 job, < 1 h | `results/container_sample_counts.txt` |

There is no run script for the explainability analysis (Fig. 5). It was done interactively in
`notebooks/06_xAI_lung.ipynb`.

## How the manuscript figures were produced

The notebooks are run from the `notebooks/` folder and read the outputs in `results/` (Zenodo archive) or `data/`.
They show how each visualization was created. Some need large inputs or API keys, as noted in the last column.

| Figure | Panel | Produced by | Based on results of | Inputs needed to re-run |
|---|---|---|---|---|
| Fig. 1 | all | overview figure, not produced by code | – | – |
| Fig. 2 | A | overview figure, not produced by code | – | – |
| Fig. 2 | B, C | `notebooks/00_llm_ontology_analysis.ipynb` (term IDs and genes per model: unique vs. shared) | `run_00` / `src/00_llm_ontologies.py` | this repository (`data/llm_ontologies/output_llms/`) |
| Fig. 2 | D | `src/05_pred_vis_eval.py` (example plot: `results/large_ontix_save/run_Aug26/large_ontix_holdout_Dim16_ClaudeOpus5_plots/Ridgeline_499_test_tissue_general.png`) | `run_06` | Zenodo archive |
| Fig. 3 | A | `notebooks/03_tuning_vis.ipynb` (hyperparameter scatter plots) | `run_04` / `src/03_large_ontix_tuning.py` | Zenodo archive (`*_tuning.pkl`) |
| Fig. 3 | B | `notebooks/05_eval_plots.ipynb` (plots A and B) | `run_06` / `src/05_pred_vis_eval.py` | Zenodo archive (`*_evaluation.csv`) |
| Fig. 4 | A | overview figure, not produced by code | – | – |
| Fig. 4 | B | `src/07_generate_with_ontix.py` (example plot: `results/large_ontix_save/run_Aug26/celltypist_comparison/celltypist_Immune_All_High_Dim32_GPT5.6_TerraPro.png`) | `run_07` | Zenodo archive |
| Fig. 4 | C | `src/07b_generate_with_varix.py` (example plot: `results/large_varix_save/run_Aug26/celltypist_comparison/celltypist_Immune_All_High_Dim32_Varix.png`) | `run_07` | Zenodo archive |
| Fig. 4 | D, E | `notebooks/07_generate_overall.ipynb` (accuracy and imbalance) | `run_07` / `src/07_generate_with_ontix.py`, `src/07b_generate_with_varix.py` | Zenodo archive (`celltypist_performance_summary_*.csv`) |
| Fig. 5 | A, B | `notebooks/06_xAI_lung.ipynb` (latent KDE plot; Captum attribution + LLM explanation, output: `notebooks/latent_explanations_lung-cancer.md`) | final model `Dim32_ClaudeOpus5` from `run_05` | holdout container (Hugging Face dataset), model `Ontix-Dim32-ClaudeOpus5`, OpenRouter API key |
| Table 1 | – | `notebooks/03_tuning_vis.ipynb`, section *Model statistics* (writes `results/model_statistics.tsv`) | `run_05` (final models) | all 15 models from Hugging Face |
| Supp. Fig. 1 | A, B | `notebooks/03_tuning_vis.ipynb` (progress over trials; ML performance vs. reconstruction loss and Spearman correlation) | `run_04` / `src/03_large_ontix_tuning.py` | Zenodo archive (`*_tuning.pkl`) |
| Supp. Fig. 2 | – | `notebooks/05_eval_plots.ipynb`, section *Supplement: detailed performance per task* | `run_06` / `src/05_pred_vis_eval.py` | Zenodo archive (`*_evaluation.csv`) |
| Supp. Fig. 3 | – | prompt template in `src/00_llm_ontologies.py` (step 1) with `data/llm_ontologies/example.json` | `run_00` | this repository (generated prompts: `data/llm_ontologies/prompts/`) |
| Supp. Table 1 | – | task definitions in `data/gemini_celltype_tasks2.json`, applied in `src/02_census_preparation.py` | `run_03` | – |

## License

Code in this repository is licensed under the Apache License 2.0 (see `LICENSE`). The corpus is derived from the
CZ CELLxGENE Census. Please follow the licenses and citation requirements of the original datasets.
