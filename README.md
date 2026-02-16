# Domain Adaptation for Protein Structure Classification
This repository contains the data collection, pre-processing pipeline, and research implementation for applying Domain Adaptation to protein structure classification (Real PDB vs. Predicted AlphaFold structures).
> The full research paper detailing our methodology, results, and analysis is available [here](https://github.com/Dev077/domain-adaptation-PDB--Alphafold/blob/main/paper/Domain%20Adaptions%20on%203D%20Protein%20Structures%20with%20AlphaFold2.pdf).
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
-----

## Architecture

> **TL;DR:** `128×128 Contact Map → 4-Layer CNN → 256-d Latent Space → Adversarial Alignment → 7-Class Prediction`

---

### 1. Input Representation

The architecture does not ingest raw 3D coordinates — it operates on a **geometric projection** of protein structure.

| Property | Value |
|---|---|
| **Tensor shape** | `(B, 1, 128, 128)` |
| **Batch size** | `B = 64` |
| **Data type** | Single-channel Float32 |

**Biological encoding pipeline:**

- Compute pairwise Euclidean distances between **Cα (Carbon-Alpha)** atoms
- Apply a binary contact threshold at **8 Å** — atom pairs closer than 8 Angstroms are marked as contacts
- Normalize variable-length proteins to `128×128` using spline interpolation via `scipy.ndimage.zoom`, preserving local topological density

The resulting contact map is a **rotation-invariant** 2D representation of the protein's tertiary structure.

---

### 2. Feature Extractor (Backbone)

A custom **4-stage CNN** trained from scratch. No ImageNet pre-training is used — features learned from natural images (cats, dogs, textures) do not transfer well to protein contact manifolds.

Each stage follows the pattern: `Conv2d → BatchNorm → ReLU → MaxPool(2)`

| Stage | Operation | Output | Biological Scale |
|---|---|---|---|
| **Conv 1** | `Conv2d(1, 32, k=3, pad=1)` | `32 × 64 × 64` | Local atomic interactions (Van der Waals) |
| **Conv 2** | `Conv2d(32, 64, k=3, pad=1)` | `64 × 32 × 32` | Secondary structures (α-helices) |
| **Conv 3** | `Conv2d(64, 128, k=3, pad=1)` | `128 × 16 × 16` | Super-secondary motifs (β-sheets, hairpins) |
| **Conv 4** | `Conv2d(128, 256, k=3, pad=1)` | `256 × 8 × 8` | Tertiary folds (global topology) |

**Bottleneck — Global Average Pooling (GAP):** Instead of flattening `256 × 8 × 8 = 16,384` parameters, GAP averages each 8×8 feature map into a single scalar, producing the final latent vector **z ∈ ℝ²⁵⁶**.

---

### 3. Fold Classifier (Task Head)

A standard MLP attached to the latent vector:

```
Linear(256, 128) → ReLU → Dropout(p=0.5) → Linear(128, 7)
```

- **Dropout at 50%** is critical — the PDB source dataset is small (~10k samples) and prone to overfitting
- Output produces softmax logits over the **7 SCOP fold classes**

---

### 4. Domain Adaptation Mechanisms

This is where the three models diverge. All operate on the shared 256-d latent space **z**.

#### A. DANN — Gradient Reversal

The encoder and classifier are **shared**. A third branch — the **domain discriminator** — is attached to z.

```
Discriminator: Linear(256, 1024) → ReLU → Dropout → Linear(1024, 1) → Sigmoid
```

The key mechanism is the **Gradient Reversal Layer (GRL)**:

| Pass | Behavior |
|---|---|
| Forward | Identity: `x → x` |
| Backward | Inverts gradient: `∇ × (−λ)` |

λ ramps from `0 → 1` during training. The encoder learns to **maximize** domain classification error (confusion), while the discriminator tries to minimize it.

#### B. ADDA — Adversarial Discriminative Domain Adaptation

The encoder weights are **decoupled**:

- **Source Encoder (Eₛ):** Pre-trained on PDB, then **frozen**
- **Target Encoder (Eₜ):** Initialized from Eₛ weights, but **learnable**

The minimax game:

- The **discriminator** tries to distinguish `Eₛ(x_pdb)` from `Eₜ(x_af)`
- The **target encoder** tries to map AlphaFold inputs into the same region of the 256-d latent space as the source encoder

Loss function: `BCEWithLogitsLoss` (standard GAN loss).

#### C. WDGRL — Wasserstein Distance Guided Representation Learning

Architecturally similar to DANN, but the domain head is a **critic** rather than a discriminator:

- **Output:** Unbounded scalar score ∈ ℝ (not a probability)
- **Constraint:** 1-Lipschitz enforced via weight clipping `[−0.01, 0.01]`
- **Update ratio:** 5 critic steps per 1 generator step

> **Note:** The aggressive weight clipping reduced the critic's capacity, leading to vanishing gradients — this is why ADDA outperformed WDGRL in the final benchmark despite WDGRL's theoretically stronger distance metric.
<img width="1054" height="821" alt="Screenshot 2026-02-16 at 1 10 30 AM" src="https://github.com/user-attachments/assets/4c0d3769-b8ae-4c0d-b4b9-12823b596dbb" />


