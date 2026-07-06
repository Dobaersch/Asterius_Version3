=============================================================================
STYLOMETRIC AUTHORSHIP VERIFICATION
(Asterius of Amasea vs. Pseudo-Chrysostom)
=============================================================================
## Project Description
This Digital Humanities project applies computational stylometry and deep learning (metric learning) to investigate the authorship of the late antique Greek sermon corpus. Specifically, it examines which of the texts transmitted anonymously or under the name of John Chrysostom (*Pseudo-Chrysostom*) exhibit a stylistic signature consistent with the authenticated works of **Asterius of Amaseia**.

By extracting high-dimensional linguistic features and training an artificial Siamese Neural Network, the pipeline calculates the Euclidean distance between anonymous texts and the calibrated author profile to establish empirically grounded authorship attributions.

---

## Methodological Pipeline

The scripts must be executed in the exact order specified below to avoid dimensional errors in the vectors:

PHASE 1: Feature Engineering of the Reference Corpus (Training)
- Script: extract_train_features.py
- Action: Extracts the raw texts of the verified authors, cleans them, splits them into 1000-token samples, and processes them with 'grc_odycy_joint_trf' (spaCy).
- Output: train_features.csv (quantified features such as MFW, POS trigrams, affixes).

PHASE 2: Model Training and Baseline Calculation
- Script: run_verification.py (and optionally validate_embeddings.py)
- Action: Trains the Siamese Network using triplet mining based on train_features.csv.
- Output: Saved model weights (.pth file), standardization scaler (.pkl file), and the established intra-author baseline.

PHASE 3: Feature Engineering of the Pseudo-Corpus (Inference)
- Script: extract_infer_features.py
- Action: Applies the exact same tokenization and extraction (restricted to the learned top features) to the Pseudo-Chrysostom corpus.
- Output: Inference CSV file with dimensions exactly matching the training matrix.

PHASE 4: Inference and Attribution
- Scripts: infer_pseudo_corpus.py (followed by aggregate_results.py)
- Action: Loads the .pth and .pkl files, calculates pairwise stylistic distances between Pseudo-samples and Asterius.
- Output: asterius_results_raw_distances.csv (sample level) and asterius_aggregated_distances.csv (document level).

PHASE 5: Visualization and Demonstration
- Script 1: plot_final_results.py (Generates diverging bar charts of the distances against the threshold).
- Script 2: plot_text_anatomy.py (Creates a stylistic text progression for borderline cases to identify compilations).
- Script 3: plot_3d_vectorspace.py (Calculates t-SNE clustering for an interactive HTML representation of the high-dimensional space).
---

## Installation & Execution

### Prerequisites
* Python 3.8 or higher
* CUDA-enabled GPU (optional, to accelerate the transformer model inference)

### Install Dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install [https://huggingface.co/chcaa/grc_odycy_joint_trf/resolve/main/grc_odycy_joint_trf-0.7.0-py3-none-any.whl](https://huggingface.co/chcaa/grc_odycy_joint_trf/resolve/main/grc_odycy_joint_trf-0.7.0-py3-none-any.whl)