# Authorship Verification: Pseudo-Chrysostom vs. Asterius of Amaseia

## Project Description
This Digital Humanities project applies computational stylometry and deep learning (metric learning) to investigate the authorship of the late antique Greek sermon corpus. Specifically, it examines which of the texts transmitted anonymously or under the name of John Chrysostom (*Pseudo-Chrysostom*) exhibit a stylistic signature consistent with the authenticated works of **Asterius of Amaseia**.

By extracting high-dimensional linguistic features and training an artificial Siamese Neural Network, the pipeline calculates the Euclidean distance between anonymous texts and the calibrated author profile to establish empirically grounded authorship attributions.

---

## Methodological Pipeline

The architecture is divided into six sequential phases:

### 1. Feature Extraction (`extract_train_features.py`)
The Ancient Greek source texts (TEI-XML or TXT) are cleaned, stripped of modern punctuation, and divided into standardized text segments (~1000 words). Utilizing the transformer-based language model `grc_odycy_joint_trf` (OdyCy), the pipeline extracts three morphosyntactic feature levels:
* Lemmatized function words (MFWs) to capture unconscious syntactic patterns.
* Part-of-Speech (POS) trigrams to map sentence structure.
* Morphological affixes to measure inflectional rhythm.

### 2. Model Training (`run_verification.py`)
The high-dimensional feature matrix (`train_features.csv`) is processed.
* **Preventing Overfitting:** The data is split into a training set (80%) and a validation set (20%), stratified by author classes.
* **Preventing Data Leakage:** The `StandardScaler` is fitted exclusively on the training set and passively transforms the validation set.
* **Architecture:** A Multi-Layer Perceptron (MLP) projects the data into a 64-dimensional embedding space, optimized via a `TripletMarginLoss` with Hard Negative Mining (control authors: Chrysostom, Severian). The script exports the learned model weights (`siamese_asterius.pth`) and the fitted scaler (`scaler.pkl`).

### 3. Vector Space Validation (`validate_embeddings.py`)
Prior to inference on unknown texts, this script loads the persisted weights and scaler to reduce the learned embedding space via PCA (global variance) and t-SNE (local neighborhoods). It generates the visual proof `embedding_validation.png`. The training is considered philologically valid if the authenticated Asterius texts form a dense, cohesive vector island, distinctly separated from the control authors.

### 4. Metric Inference (`infer_pseudo_corpus.py`)
The unlabeled Pseudo-Chrysostom corpus is projected into the calibrated vector space.
* The script calculates the exact geometric center (centroid) of the Asterius style.
* It defines a **dynamic intra-author baseline (threshold)** based on the maximum internal deviation of the authenticated Asterius dataset.
* Each anonymous text segment is automatically classified as: *Core-Asterius* (within the threshold), *Gray Zone* (theoretical tolerance margin for genre noise), or *Rejected*.

### 5. Document Aggregation (`aggregate_results.py`)
Because the inference operates on 1000-word samples for statistical stability, this script handles the post-processing. Using regular expressions, it aggregates the segmented results back to the document level. It calculates the **Asterius Match Percentage** (the percentage of a text's segments that fall within the Asterius cluster) and exports the final synthesis `asterius_final_document_scores.csv`.

### 6. Publication Visualization (`plot_final_results.py`)
This script processes the aggregated data for academic publication. It generates a *Diverging Bar Chart* (`attribution_diverging_bar.png`) in high print quality (300 DPI), visually juxtaposing the mean Euclidean distance of the relevant texts against the critical red demarcation line of the intra-author threshold.

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