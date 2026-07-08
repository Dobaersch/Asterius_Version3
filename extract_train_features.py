import os
import re
import json
import pandas as pd
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


def extract_train_features():
    # --- Configuration ---
    TRAIN_BASE_DIR = "data/train"
    BIBLE_PATH = "greek_bible.txt"
    OUTPUT_CSV = "train_features.csv"
    VOCAB_JSON = "top_features_vocabulary.json"

    print("--- Starting Feature Extraction (Training Corpus) ---")

    vectorizer, bible_tfidf = build_bible_vectorizer(BIBLE_PATH)

    def is_bible_quote(sentence_text, threshold=0.85):
        if not vectorizer or len(sentence_text) < 15:
            return False
        s_vec = vectorizer.transform([sentence_text])
        return cosine_similarity(s_vec, bible_tfidf).max() >= threshold

    # Dictionaries to capture corpus-wide frequencies for our dynamic MFW logic
    global_counts = {
        'words': Counter(),
        'pos': Counter(),
        'morph': Counter()
    }

    sample_records = []

    if not os.path.exists(TRAIN_BASE_DIR):
        raise FileNotFoundError(f"Training directory '{TRAIN_BASE_DIR}' does not exist.")

    authors = [d for d in os.listdir(TRAIN_BASE_DIR) if os.path.isdir(os.path.join(TRAIN_BASE_DIR, d))]

    for author in authors:
        author_dir = os.path.join(TRAIN_BASE_DIR, author)
        print(f"\n[Processing] Author: {author.capitalize()}")

        for filename in os.listdir(author_dir):
            if filename.startswith('.'):
                continue

            file_path = os.path.join(author_dir, filename)
            try:
                raw_text = read_file_safely(file_path)
                raw_text = clean_text(raw_text, filename)

                if len(raw_text) < 50:
                    continue

                doc = nlp(raw_text)
            except Exception as e:
                print(f"  [Error] Skipping file '{filename}': {e}")
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
                            global_counts['words'][lemma] += 1

                        # 2. Morphological Features
                        if token.morph:
                            morph_str = str(token.morph)
                            current_m[morph_str] = current_m.get(morph_str, 0) + 1
                            global_counts['morph'][morph_str] += 1

                        # 3. Syntactic POS Trigrams
                        children = sorted(list(token.children), key=lambda c: c.i)
                        if len(children) >= 2:
                            for i in range(len(children) - 1):
                                trigram = (children[i].pos_, token.pos_, children[i + 1].pos_)
                                current_syntactic_trigrams.append(trigram)
                                global_counts['pos'][f"{trigram[0]}_{trigram[1]}_{trigram[2]}"] += 1

                # Rolling Window threshold mathematically fixed to >= 1000
                if current_length >= 1000:
                    sample_records.append({
                        "author": author.capitalize(),
                        "title": f"{filename}_{chunk_index}",
                        "w": dict(current_w),
                        "p": dict(Counter(current_syntactic_trigrams)),
                        "m": dict(current_m)
                    })

                    # Reset memory for the next chunk
                    current_w, current_m, current_syntactic_trigrams = {}, {}, []
                    current_length = 0
                    chunk_index += 1

            # Process the remaining tail chunk
            if current_length >= 1000 or (chunk_index == 0 and current_length >= 250):
                sample_records.append({
                    "author": author.capitalize(),
                    "title": f"{filename}_{chunk_index}",
                    "w": dict(current_w),
                    "p": dict(Counter(current_syntactic_trigrams)),
                    "m": dict(current_m)
                })

    print("\n--- Step 2: Global Vocabulary & MFW Calculation ---")

    # 1. Retrieve the Top 150 Most Frequent Words dynamically
    MFW_COUNT = 150
    top_words = [w for w, _ in global_counts['words'].most_common(MFW_COUNT)]

    # 2. Retrieve Top 100 Syntactic POS Trigrams
    top_pos = [p for p, _ in global_counts['pos'].most_common(100)]

    # 3. Retrieve Top 100 Morphological features
    top_morph = [m for m, _ in global_counts['morph'].most_common(100)]

    # Export the dynamically calculated vocabulary for the inference step
    with open(VOCAB_JSON, 'w', encoding='utf-8') as f:
        json.dump({'words': top_words, 'pos': top_pos, 'morph': top_morph}, f)

    print(f"[Success] Dynamic vocabulary exported to '{VOCAB_JSON}'")

    print("\n--- Step 3: Formatting Final Matrix ---")
    final_rows = []

    for r in sample_records:
        row = {"Auteur": r["author"], "Titre": r["title"]}

        for w in top_words:
            row[f"LEMMA_{w}"] = r["w"].get(w, 0)

        p_c = r["p"]
        pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
        for p in top_pos:
            row[f"POS_{p}"] = pos_dict.get(p, 0)

        for m in top_morph:
            row[f"MORPH_{m}"] = r["m"].get(m, 0)

        final_rows.append(row)

    df_train = pd.DataFrame(final_rows).fillna(0)
    df_train.to_csv(OUTPUT_CSV, index=False)

    print(f"[Success] Training features formatted and saved to '{OUTPUT_CSV}'")
    print(f"[Info] Shape of the training matrix: {df_train.shape}")


if __name__ == "__main__":
    extract_train_features()