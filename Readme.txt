##Step 1: Environment Initialization
Install all core libraries and pull the domain-specific transformer model for Ancient Greek (OdyCy) from HuggingFace.

Bash
pip install -r requirements.txt
pip install https://huggingface.co/chcaa/grc_odycy_joint_trf/resolve/main/grc_odycy_joint_trf-0.7.0-py3-none-any.whl

##Step 2: Extract Training Features
Run extract_train_features.py. This script processes the known corpora, isolates the most frequent lemmatized functional words, POS trigrams, and morphological tags. It establishes the master feature space and exports top_features_vocabulary.json alongside the primary dataset train_features.csv.

Bash
python extract_train_features.py
Methodological Note: Texts are sliced into standardized 1000-word tokens to normalize frequency distributions across unevenly transmitted sermons.

##Step 3: Align Inference Features
Run extract_infer_features.py. This script processes the target Pseudo-Chrysostom corpus. Crucially, it does not calculate a new vocabulary. Instead, it forces the anonymous data into the exact dimensional shape of the training feature space by reading the previously saved top_features_vocabulary.json.

Bash
python extract_infer_features.py
Methodological Note: This isolates the inference model from shape mismatch errors and prevents critical data leakage.

##Step 4: Model Training & Hard Negative Mining
Run run_verification.py to train the Siamese Multi-Layer Perceptron (MLP). The dataset loader pairs authentic Asterius tokens against authentic Asterius tokens (positives) and leverages a 70% bias toward Chrysostom/Severian texts (Hard Negatives) during Triplet Generation.

Bash
python run_verification.py
This script saves the optimal weights to siamese_asterius.pth and dumps the fitted StandardScaler to scaler.pkl.

##Step 5: Latent Space Validation
Before running attribution, execute validate_embeddings.py to visually inspect the geometry of the newly defined 64-dimensional latent embedding space. It applies PCA and t-SNE dimensionality reduction.

Bash
python validate_embeddings.py
Validation Criterion: The script outputs embedding_validation.png. Authentic Asterius tokens must form a highly cohesive, distinct cluster clearly separated from both the true Chrysostom group and the other patristic prose samples. If clusters overlap chaotically, adjust hyperparameters or increase slice token lengths before proceeding.

##Step 6: Metric Inference & Centroid Attribution
Run infer_pseudo_corpus.py. The script automatically reconstructs the ideal signature profile of Asterius by calculating his geometric mathematical mean (Centroid) across all known authentic embeddings. It calculates an intra-author baseline threshold (the maximum distance an authentic text has to its own centroid).

Bash
python infer_pseudo_corpus.py
It projects the anonymous inference_features.csv data into this space, computes Euclidean distances to the Asterius Centroid, and logs any entry that falls safely inside the threshold.

#Interpreting Output Data
The pipeline generates asterius_candidates.csv, sorted in ascending order by distance to the Asterius profile:

Distanz_zu_Asterius: Lower values indicate a closer micro-stylistic affinity.

Klassifikation: Explicitly flags samples as Asterius or Spuria (Fremd).

Konfidenz_%: Represents how deeply a text sits within the empirical boundaries of the authentic intra-author threshold.