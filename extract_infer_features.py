import os
import re
import pandas as pd
import json
import spacy
from bs4 import BeautifulSoup
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

    # If no bible reference is loaded or sentence is too short, return all-False mask
    if not bible_ngrams or len(sent) < 8:
        return mask

    # Extract normalized strings directly from spaCy tokens to maintain strict index alignment
    norm_tokens = [strip_greek_diacritics(t.text) for t in sent]

    temp_mask = [False] * len(sent)
    for i in range(len(norm_tokens) - 3):
        if tuple(norm_tokens[i:i + 4]) in bible_ngrams:
            temp_mask[i] = True
            temp_mask[i + 1] = True
            temp_mask[i + 2] = True
            temp_mask[i + 3] = True

    matching_words = sum(temp_mask)
    # We still use the threshold of 8 words to prevent accidental common 4-gram overlaps
    if matching_words >= 8:
        print(f"  [Filtered] Masked {matching_words} biblical tokens in sentence.")
        return temp_mask

    return mask


def main():
    print("--- Starting Inference Feature Extraction ---")

    # 1. Load Vocabulary
    vocab_path = "top_features_vocabulary.json"
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file '{vocab_path}' not found.")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    top_words = vocab["top_words"]
    top_pos = vocab["top_pos"]
    top_morph = vocab["top_morph"]

    print(f"Loaded Vocabulary: {len(top_words)} words, {len(top_pos)} POS, {len(top_morph)} morphs.")

    # 2. Load Bible Reference
    bible_path = "greek_bible.txt"
    bible_ngrams = set()
    if os.path.exists(bible_path):
        bible_text = read_file_safely(bible_path)
        bible_words = [strip_greek_diacritics(w) for w in bible_text.split()]
        # Generate 4-grams for the bible reference
        bible_ngrams = {tuple(bible_words[i:i + 4]) for i in range(len(bible_words) - 3)}
        print(f"Loaded Bible reference with {len(bible_ngrams)} 4-grams.")
    else:
        print(f"Warning: Bible reference '{bible_path}' not found. Bible filtering will be skipped.")

    inference_dir = "data/inference/pseudo_corpus"
    sample_records = []

    print("\n--- Step 1: Processing Inference Corpus ---")

    if not os.path.exists(inference_dir):
        print(f"Error: Inference directory '{inference_dir}' not found.")
        return

    for file in os.listdir(inference_dir):
        if file.endswith(('.xml', '.txt')):
            file_path = os.path.join(inference_dir, file)
            raw_text = read_file_safely(file_path)

            # Clean HTML/XML tags if applicable
            if file.endswith('.xml'):
                soup = BeautifulSoup(raw_text, 'xml')
                text = soup.get_text(separator=' ')
            else:
                text = raw_text

            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue

            print(f"Processing: {file}")
            doc = nlp(text)

            current_w = {}
            current_p = {}
            current_m = {}
            current_length = 0

            for sent in doc.sents:
                quote_mask = get_bible_quote_mask(sent, bible_ngrams)

                for i, token in enumerate(sent):
                    # Skip tokens that are flagged as part of a biblical quote
                    if quote_mask[i]:
                        continue

                    # Only increment rolling window length for actual authorial text!
                    current_length += 1

                    # 1. Dynamic Extraction of all alphabet tokens
                    if token.is_alpha:
                        lemma = token.lemma_.lower()
                        current_w[lemma] = current_w.get(lemma, 0) + 1

                    # 2. Morphological Features
                    if token.morph:
                        morph_str = str(token.morph)
                        current_m[morph_str] = current_m.get(morph_str, 0) + 1

                    # 3. Syntactic POS Trigrams
                    # Ensure dependencies (children) do not cross into masked biblical quotes.
                    valid_children = [
                        c for c in token.children
                        if c.i >= sent.start and c.i < sent.end and not quote_mask[c.i - sent.start]
                    ]
                    valid_children = sorted(valid_children, key=lambda c: c.i)

                    if len(valid_children) >= 2:
                        for j in range(len(valid_children) - 1):
                            trigram = f"{valid_children[j].pos_}_{token.pos_}_{valid_children[j + 1].pos_}"
                            current_p[trigram] = current_p.get(trigram, 0) + 1

                # --- ROLLING WINDOW ---
                if current_length >= 1000:
                    sample_records.append({
                        "author": "Pseudo-Chrysostomus",
                        "title": file.replace('.xml', '').replace('.txt', ''),
                        "w": current_w,
                        "p": current_p,
                        "m": current_m
                    })
                    current_w = {}
                    current_p = {}
                    current_m = {}
                    current_length = 0

            # Process remaining text (Tail-Chunk) if it meets the minimum threshold
            if current_length >= 250:
                sample_records.append({
                    "author": "Pseudo-Chrysostomus",
                    "title": file.replace('.xml', '').replace('.txt', ''),
                    "w": current_w,
                    "p": current_p,
                    "m": current_m
                })

    print("\n--- Step 2: Formatting Final Inference Matrix (Relative Frequencies) ---")
    final_rows = []

    for r in sample_records:
        row = {"Auteur": r["author"], "Titre": r["title"]}

        # Calculate total valid tokens/features in this specific chunk to normalize frequencies
        total_tokens = sum(r["w"].values()) if sum(r["w"].values()) > 0 else 1
        total_pos = sum(r["p"].values()) if sum(r["p"].values()) > 0 else 1
        total_morph = sum(r["m"].values()) if sum(r["m"].values()) > 0 else 1

        for w in top_words:
            # Multiply by 1000 to get normalized rates per 1000 tokens
            row[f"LEMMA_{w}"] = (r["w"].get(w, 0) / total_tokens) * 1000

        for p in top_pos:
            row[f"POS_{p}"] = (r["p"].get(p, 0) / total_pos) * 1000

        for m in top_morph:
            row[f"MORPH_{m}"] = (r["m"].get(m, 0) / total_morph) * 1000

        final_rows.append(row)

    df = pd.DataFrame(final_rows)
    # Fill remaining NaNs with 0 to ensure matrix integrity
    df = df.fillna(0)

    output_path = "inference_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[Success] Inference features saved to {output_path}. Shape: {df.shape}")


if __name__ == "__main__":
    main()