# Domain Adaptation for Protein Structure Classification
This repository contains the data collection, pre-processing pipeline, and research implementation for applying Domain Adaptation to protein structure classification (Real PDB vs. Predicted AlphaFold structures).
##  Data Availability
**Important:** To maintain a lightweight repository, the `data/` folder **only contains the final processed features** required to train the models:
  * `data/features/pdb_maps.npy`
  * `data/features/af_maps.npy`
  * `data/features/labels.npy`
  * `data/features/ids.npy`
The raw data (PDB files, AlphaFold predictions, and FASTA sequences) are **not included**. If you wish to regenerate the dataset from scratch, please follow the Data Pipeline instructions below.
-----
##  Research Implementation & Models
The core model training, validation, and testing are contained within the `research_implementation.ipynb` notebook.
### Execution Environment
  * **Platform:** Google Colab
  * **Hardware:** Trained using **L4 / A100 / T4 GPUs**.
  * **Dependencies:** PyTorch, Scikit-Learn, NumPy, Matplotlib.
### Implemented Models
The notebook implements and executes the following **4 models** to evaluate domain adaptation performance:
1.  **Naive (Baseline):** Trained on PDB data, tested on AlphaFold data with no domain adaptation.
2.  **Gradient Reversal (DANN):** Uses a gradient reversal layer to learn domain-invariant features through adversarial training.
3.  **ADDA:** Adversarial Discriminative Domain Adaptation with separate source and target encoders.
4.  **WDGRL:** Wasserstein Distance Guided Representation Learning for stable domain alignment.
> To reproduce the results, open the notebook in Google Colab, upload the processed `.npy` files to the session storage (or mount Drive), and run all cells.
-----
##  Data Collection & Processing Pipeline
If you need to regenerate the features from scratch, execute the following scripts **in this exact order**.
### 1\. Parse Classifications
  * **Script:** `parse_scop.py`
  * **Action:** Parses raw SCOP text files to extract class hierarchies.
  * **Output:** `data/scop/parsed_classifications.csv`
### 2\. Stratified Sampling
  * **Script:** `create_stratified_dataset.py`
  * **Action:** Balances the dataset by selecting \~2000 samples per class to ensure uniform distribution.
  * **Output:** `data/scop/sampled_dataset.csv`
### 3\. Download ASTRAL Database
  * **Script:** `map_sequences.py`
  * **Action:** Downloads the massive `astral-all.fa` file required for sequence mapping.
  * **Output:** `astral-all.fa`
### 4\. Sequence Mapping
  * **Script:** `create_sampled_data_with_seq.py`
  * **Action:** Maps the correct amino acid sequences from ASTRAL to the sampled SCOP IDs.
  * **Output:** `sampled_dataset_with_seqs.csv`
### 5\. Download Experimental Structures (PDB)
  * **Script:** `download_PDB.py`
  * **Action:** Downloads legacy `.ent` structure files for the source domain.
  * **Output:** `data/pdb/*.ent`
  * *Note: Ensure the script reads the correct input filename `sampled_dataset_with_seqs.csv`).*
### 6\. Download Predicted Structures (AlphaFold)
  * **Script:** `download_alphafold.py`
  * **Action:** Fetches predicted structures via the AlphaFold API and UniProt mapping.
  * **Output:** `data/alphafold/*.pdb` and `mappingPDBtoUniProt.csv`
### 7\. Feature Extraction (Contact Maps)
  * **Script:** `contact_arrays.py`
  * **Action:** Aligns sequences, extracts 3D coordinates, generates binary contact maps, and resizes them to 128x128.
  * **Output:** `data/features/*.npy` (The final inputs for the models).
### 8\. Verification
  * **Script:** `data_verifyication.py`
  * **Action:** Checks array shapes and visualizes the domain shift between PDB and AlphaFold maps.
  * **Output:** Console stats and Matplotlib plots.
