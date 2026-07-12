import os
import re
import json
import pandas as pd
import numpy as np
import spacy
from bs4 import BeautifulSoup
from sklearn.feature_selection import f_classif
import unicodedata
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- NLP Setup ---
spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")
nlp.max_length = 3000000

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def strip_greek_diacritics(text):
    """
    Removes polytonic accents and breathings for pure character matching.
    Converts to lowercase to ensure absolute baseline string comparison.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower()


def read_file_safely(file_path):
    """Safely decode Greek texts with multiple fallbacks."""
    encodings = ['utf-8-sig', 'utf-8', 'iso-8859-7', 'windows-1253', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to decode {file_path}")


def get_bible_quote_mask(sent, bible_ngrams):
    """
    Creates a boolean mask for the tokens in a spaCy sentence.
    Returns a list of booleans (True = token is part of a Bible quote).
    """
    mask = [False] * len(sent)

    if not bible_ngrams or len(sent) < 8:
        return mask

    norm_tokens = [strip_greek_diacritics(t.text) for t in sent]
    temp_mask = [False] * len(sent)

    for i in range(len(norm_tokens) - 3):
        if tuple(norm_tokens[i:i + 4]) in bible_ngrams:
            temp_mask[i] = True
            temp_mask[i + 1] = True
            temp_mask[i + 2] = True
            temp_mask[i + 3] = True

    matching_words = sum(temp_mask)
    if matching_words >= 8:
        print(f"  [Filtered] Masked {matching_words} biblical tokens in sentence.")
        return temp_mask

    return mask


def main():
    print("--- Starting Train Feature Extraction ---")

    # 1. Load Bible Reference
    bible_path = "greek_bible.txt"
    bible_ngrams = set()
    if os.path.exists(bible_path):
        bible_text = read_file_safely(bible_path)
        bible_words = [strip_greek_diacritics(w) for w in bible_text.split()]
        bible_ngrams = {tuple(bible_words[i:i + 4]) for i in range(len(bible_words) - 3)}
        print(f"Loaded Bible reference with {len(bible_ngrams)} 4-grams.")
    else:
        print(f"Warning: Bible reference '{bible_path}' not found. Bible filtering will be skipped.")

    # Dynamically find all author directories in data/train
    train_base_dir = "data/train"
    if not os.path.exists(train_base_dir):
        print(f"Error: Base training directory '{train_base_dir}' not found.")
        return

    train_dirs = [os.path.join(train_base_dir, d) for d in os.listdir(train_base_dir)
                  if os.path.isdir(os.path.join(train_base_dir, d))]

    sample_records = []
    global_counts = {'words': {}, 'pos': {}, 'morph': {}}

    print("\n--- Step 1: Processing Training Corpus ---")

    for d in train_dirs:
        author_label = os.path.basename(d)

        for file in os.listdir(d):
            if file.endswith(('.xml', '.txt')):
                file_path = os.path.join(d, file)
                raw_text = read_file_safely(file_path)

                if file.endswith('.xml'):
                    soup = BeautifulSoup(raw_text, 'xml')
                    text = soup.get_text(separator=' ')
                else:
                    text = raw_text

                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    continue

                print(f"Processing: [{author_label}] {file}")
                doc = nlp(text)

                current_w = {}
                current_p = {}
                current_m = {}
                current_length = 0

                for sent in doc.sents:
                    quote_mask = get_bible_quote_mask(sent, bible_ngrams)

                    for i, token in enumerate(sent):
                        if quote_mask[i]:
                            continue

                        current_length += 1

                        # 1. Lemmas
                        if token.is_alpha:
                            lemma = token.lemma_.lower()
                            current_w[lemma] = current_w.get(lemma, 0) + 1
                            global_counts['words'][lemma] = global_counts['words'].get(lemma, 0) + 1

                        # 2. Morphology
                        if token.morph:
                            morph_str = str(token.morph)
                            current_m[morph_str] = current_m.get(morph_str, 0) + 1
                            global_counts['morph'][morph_str] = global_counts['morph'].get(morph_str, 0) + 1

                        # 3. POS Trigrams
                        valid_children = [
                            c for c in token.children
                            if c.i >= sent.start and c.i < sent.end and not quote_mask[c.i - sent.start]
                        ]
                        valid_children = sorted(valid_children, key=lambda c: c.i)

                        if len(valid_children) >= 2:
                            for j in range(len(valid_children) - 1):
                                trigram = f"{valid_children[j].pos_}_{token.pos_}_{valid_children[j + 1].pos_}"
                                current_p[trigram] = current_p.get(trigram, 0) + 1
                                global_counts['pos'][trigram] = global_counts['pos'].get(trigram, 0) + 1

                # --- ROLLING WINDOW ---
                if current_length >= 1000:
                    sample_records.append({
                        "author": author_label,
                        "title": file.replace('.xml', '').replace('.txt', ''),
                        "w": current_w,
                        "p": current_p,
                        "m": current_m
                    })
                    current_w = {}
                    current_p = {}
                    current_m = {}
                    current_length = 0

            # Tail-Chunk Export
            if current_length >= 250:
                sample_records.append({
                    "author": author_label,
                    "title": file.replace('.xml', '').replace('.txt', ''),
                    "w": current_w,
                    "p": current_p,
                    "m": current_m
                })

    print("\n--- Step 2: Selecting Top Features via ANOVA F-Value ---")

    # 1. Sort global counts by absolute frequency
    words_sorted = sorted(global_counts['words'].items(), key=lambda x: x[1], reverse=True)
    top_words = [k for k, v in words_sorted[:200]]

    morph_sorted = sorted(global_counts['morph'].items(), key=lambda x: x[1], reverse=True)
    top_morph = [k for k, v in morph_sorted[:100]]

    # 2. Advanced Feature Selection for POS Trigrams (Filtering generic syntax)
    pos_sorted = sorted(global_counts['pos'].items(), key=lambda x: x[1], reverse=True)
    candidate_pos = [k for k, v in pos_sorted[:300]]

    print(f"  [Info] Running ANOVA F-Test on {len(candidate_pos)} frequent POS trigrams to filter generic syntax...")

    temp_pos_matrix = []
    labels = []
    for r in sample_records:
        labels.append(r["author"])
        total_pos_in_chunk = sum(r["p"].values()) if sum(r["p"].values()) > 0 else 1

        row_vals = []
        for cp in candidate_pos:
            val = (r["p"].get(cp, 0) / total_pos_in_chunk) * 1000
            row_vals.append(val)
        temp_pos_matrix.append(row_vals)

    X_pos_temp = np.array(temp_pos_matrix)
    y_labels = np.array(labels)

    f_values, p_values = f_classif(X_pos_temp, y_labels)

    scored_pos = list(zip(candidate_pos, f_values))
    scored_pos = [(p, score) for p, score in scored_pos if not np.isnan(score)]
    scored_pos.sort(key=lambda x: x[1], reverse=True)

    top_pos = [p for p, score in scored_pos[:100]]
    print(f"  [Result] Selected {len(top_pos)} highly author-specific POS trigrams.")

    # 3. Save the final optimized vocabulary
    vocab_path = "top_features_vocabulary.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'top_words': top_words, 'top_pos': top_pos, 'top_morph': top_morph}, f)
    print(f"Saved vocabulary ({len(top_words)} words, {len(top_pos)} POS, {len(top_morph)} morphs) to {vocab_path}")

    print("\n--- Step 3: Formatting Final Training Matrix (Relative Frequencies) ---")
    final_rows = []

    for r in sample_records:
        row = {"Auteur": r["author"], "Titre": r["title"]}

        total_tokens = sum(r["w"].values()) if sum(r["w"].values()) > 0 else 1
        total_pos = sum(r["p"].values()) if sum(r["p"].values()) > 0 else 1
        total_morph = sum(r["m"].values()) if sum(r["m"].values()) > 0 else 1

        for w in top_words:
            row[f"LEMMA_{w}"] = (r["w"].get(w, 0) / total_tokens) * 1000

        for p in top_pos:
            row[f"POS_{p}"] = (r["p"].get(p, 0) / total_pos) * 1000

        for m in top_morph:
            row[f"MORPH_{m}"] = (r["m"].get(m, 0) / total_morph) * 1000

        final_rows.append(row)

    df = pd.DataFrame(final_rows)
    df = df.fillna(0)

    output_path = "train_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[Success] Training features saved to {output_path}. Shape: {df.shape}")


if __name__ == "__main__":
    main()