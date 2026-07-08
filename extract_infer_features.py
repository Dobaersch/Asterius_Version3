import os
import re
import pandas as pd
import json
from collections import Counter
import spacy
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- NLP Setup ---
spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")
nlp.max_length = 3000000

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def read_file_safely(file_path):
    """Safely decode Greek texts with multiple fallbacks."""
    encodings = ['utf-8-sig', 'utf-8', 'iso-8859-7', 'windows-1253', 'latin-1']
    last_error = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise ValueError(f"Could not decode file {file_path}. Last error: {last_error}")


def clean_text(text, filename):
    """Parses XML cleanly, deletes TEI-Headers and removes conjectures from TXT."""
    if filename.lower().endswith('.xml'):
        text = re.sub(r'<\?xml.*?\?>', '', text).strip()
        wrapped_text = f"<document>{text}</document>"
        try:
            soup = BeautifulSoup(wrapped_text, "xml")
            for header in soup.find_all('teiHeader'):
                header.decompose()
            text = soup.get_text(separator=' ')
        except Exception:
            text = re.sub(r'<[^>]+>', ' ', text)
    else:
        text = text.replace('<', '').replace('>', '')

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_bible_vectorizer(bible_path="greek_bible.txt"):
    """Builds a TF-IDF character-trigram space of the Septuagint/NT to filter out citations."""
    if not os.path.exists(bible_path):
        print(f"[Warning] Bible reference file '{bible_path}' not found. Quotes will not be filtered.")
        return None, None
    with open(bible_path, 'r', encoding='utf-8', errors='ignore') as f:
        bible_text = f.read()

    bible_text = re.sub(r'\s+', ' ', bible_text).strip()
    bible_sentences = re.split(r'[.·;]+', bible_text)
    bible_sentences = [s.strip() for s in bible_sentences if len(s.strip()) > 10]

    if not bible_sentences:
        return None, None

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    bible_tfidf = vectorizer.fit_transform(bible_sentences)
    return vectorizer, bible_tfidf


def extract_inference_features():
    # --- Configuration ---
    INFERENCE_FOLDER = "data/inference/pseudo_corpus"
    BIBLE_PATH = "greek_bible.txt"
    VOCAB_JSON = "top_features_vocabulary.json"
    OUTPUT_CSV = "inference_features.csv"

    print("--- Starting Feature Extraction (Inference Corpus) ---")

    if not os.path.exists(VOCAB_JSON):
        raise FileNotFoundError(
            f"[Critical Error] Vocabulary '{VOCAB_JSON}' is missing. Run training extraction first.")

    # Load dynamic vocabulary from the training phase
    with open(VOCAB_JSON, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        top_words = vocab['words']
        top_pos = vocab['pos']
        top_morph = vocab['morph']

    vectorizer, bible_tfidf = build_bible_vectorizer(BIBLE_PATH)

    def is_bible_quote(sentence_text, threshold=0.85):
        if not vectorizer or len(sentence_text) < 15:
            return False
        s_vec = vectorizer.transform([sentence_text])
        return cosine_similarity(s_vec, bible_tfidf).max() >= threshold

    sample_records = []

    if not os.path.exists(INFERENCE_FOLDER):
        raise FileNotFoundError(f"[Critical Error] Inference directory '{INFERENCE_FOLDER}' does not exist.")

    for filename in os.listdir(INFERENCE_FOLDER):
        if filename.startswith('.'):
            continue

        file_path = os.path.join(INFERENCE_FOLDER, filename)
        try:
            raw_text = read_file_safely(file_path)
            raw_text = clean_text(raw_text, filename)

            if len(raw_text) < 50:
                print(f"[Skipped] '{filename}' contains insufficient text.")
                continue

            doc = nlp(raw_text)
        except Exception as e:
            print(f"[Error] Skipping file '{filename}': {e}")
            continue

        current_w = {}
        current_m = {}
        current_syntactic_trigrams = []

        current_length = 0
        chunk_index = 0

        for sent in doc.sents:
            if not is_bible_quote(sent.text):
                current_length += len(sent)

                for token in sent:
                    # 1. Dynamic Extraction of all alphabet tokens
                    if token.is_alpha:
                        lemma = token.lemma_.lower()
                        current_w[lemma] = current_w.get(lemma, 0) + 1

                    # 2. Morphological Features
                    if token.morph:
                        morph_str = str(token.morph)
                        current_m[morph_str] = current_m.get(morph_str, 0) + 1

                    # 3. Syntactic POS Trigrams
                    children = sorted(list(token.children), key=lambda c: c.i)
                    if len(children) >= 2:
                        for i in range(len(children) - 1):
                            trigram = (children[i].pos_, token.pos_, children[i + 1].pos_)
                            current_syntactic_trigrams.append(trigram)

        # --- ROLLING WINDOW ---
        if current_length >= 1000:
            p_c = Counter(current_syntactic_trigrams)
            row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}

            # Map dynamically to the established feature space
            for w in top_words: row[f"LEMMA_{w}"] = current_w.get(w, 0)
            pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
            for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
            for m in top_morph: row[f"MORPH_{m}"] = current_m.get(m, 0)

            sample_records.append(row)

            # Reset memory for the next chunk
            current_w, current_m, current_syntactic_trigrams = {}, {}, []
            current_length = 0
            chunk_index += 1

        # Process the remaining tail chunk
        if current_length >= 1000 or (chunk_index == 0 and current_length >= 250):
            if chunk_index == 0 and current_length < 1000:
                print(
                    f"[Warning] Text '{filename}' is very short ({current_length} tokens). Results may be statistically unstable.")

            p_c = Counter(current_syntactic_trigrams)
            row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}

            for w in top_words: row[f"LEMMA_{w}"] = current_w.get(w, 0)
            pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
            for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
            for m in top_morph: row[f"MORPH_{m}"] = current_m.get(m, 0)

            sample_records.append(row)

    print("\n--- Step 2: Formatting Final Inference Matrix ---")
    df_infer = pd.DataFrame(sample_records).fillna(0)
    df_infer.to_csv(OUTPUT_CSV, index=False)

    print(f"[Success] Inference features formatted and saved to '{OUTPUT_CSV}'")
    print(f"[Info] Shape of the inference matrix: {df_infer.shape}")


if __name__ == "__main__":
    extract_inference_features()