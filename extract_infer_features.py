import os
import re
import pandas as pd
import json
import numpy as np
from collections import Counter
from nltk.util import ngrams
import spacy
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP-Modell laden und Sentencizer sicherstellen
spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")
nlp.max_length = 2000000

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Globale Konstanten
GREEK_FUNCTION_WORDS_LEMMATA = {
    "καί", "δέ", "τε", "ἀλλά", "ἤ", "γάρ", "οὖν", "ἄρα", "διό",
    "ἵνα", "ὅπως", "ὡς", "ὥστε", "ὅτι", "εἰ", "ἐάν", "ἐπεί", "ἐπειδή",
    "οὔτε", "μήτε", "οὐδέ", "μηδέ", "πλήν", "ἐν", "εἰς", "ἐκ", "ἐξ",
    "πρός", "ἐπί", "διά", "κατά", "μετά", "παρά", "ἀπό", "ὑπέρ", "ὑπό",
    "περί", "ἀντί", "πρό", "σύν", "ἄνευ", "ἕνεκα", "μέχρι", "ἄχρι",
    "μέν", "ἄν", "γε", "δή", "τοι", "που", "πως", "ποτέ", "ἔτι",
    "ἤδη", "νῦν", "οὕτως", "ὧδε", "πάνυ", "μάλιστα", "οὐ", "μή",
    "οὐκ", "οὐχ", "οὐχί", "μηκέτι", "οὐκέτι", "ὁ", "ἡ", "τό",
    "ἐγώ", "σύ", "αὐτός", "ὅς", "ὅστις", "οὗτος", "ἐκεῖνος",
    "ὅδε", "τίς", "τις", "ἑαυτοῦ", "ἀλλήλων", "τοιοῦτος", "τοσοῦτος",
    "εἰμί", "γίγνομαι", "ἔχω"
}

ELISION_MAP = {
    "ἀλλ'": "ἀλλά", "δι'": "διά", "κατ'": "κατά", "μεθ'": "μετά",
    "μετ'": "μετά", "παρ'": "παρά", "ἐπ'": "ἐπί", "ἐφ'": "ἐπί",
    "ὑπ'": "ὑπό", "ὑφ'": "ὑπό", "ἀπ'": "ἀπό", "ἀφ'": "ἀπό",
    "ἀντ'": "ἀντί", "ἀνθ'": "ἀντί", "οὐκ": "οὐ", "οὐχ": "οὐ",
    "τ'": "τε", "δ'": "δέ", "γ'": "γε", "μ'": "με"
}


# --- Zitat-Filterungs-Funktionen ---
def load_bible_reference(filepath="greek_bible.txt"):
    print(f"Lade Bibel-Referenzkorpus für Zitat-Filterung aus '{filepath}'...")
    if not os.path.exists(filepath):
        print(f"[Warnung] Bibel-Referenzdatei '{filepath}' nicht gefunden. Zitat-Filterung wird übersprungen.")
        return None, None

    with open(filepath, 'r', encoding='utf-8') as f:
        bible_verses = f.readlines()

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
    bible_matrix = vectorizer.fit_transform(bible_verses)
    print("Bibel-Referenzkorpus erfolgreich geladen und vektorisiert.")
    return vectorizer, bible_matrix


def is_bible_quote(sentence_text, vectorizer, bible_matrix, threshold=0.45):
    if vectorizer is None or bible_matrix is None:
        return False

    if len(sentence_text.split()) < 4:
        return False

    sent_vec = vectorizer.transform([sentence_text])
    max_sim = np.max(cosine_similarity(sent_vec, bible_matrix))

    return max_sim > threshold


# ----------------------------------------

def extract_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = BeautifulSoup(f, 'lxml-xml').get_text(separator=' ') if filepath.endswith(".xml") else f.read()

    clean_text = re.sub(r'[^\u0370-\u03FF\u1F00-\u1FFF\s\.,;··!\?]', ' ', raw_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text.strip()


def normalize_greek_token(token):
    lemma = token.lemma_.lower() if token.lemma_ else token.text.lower()
    text_lower = token.text.lower()

    if text_lower in ELISION_MAP:
        return ELISION_MAP[text_lower]

    return lemma


def process_inference_corpus(input_dir, output_csv, vocab_json):
    if not os.path.exists(vocab_json):
        raise FileNotFoundError(f"Vokabular-Datei '{vocab_json}' fehlt. Führe erst extract_train_features.py aus!")

    with open(vocab_json, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    top_words = vocab['words']
    top_pos = vocab['pos']
    top_morph = vocab['morph']

    bible_vectorizer, bible_matrix = load_bible_reference("greek_bible.txt")

    sample_records = []
    valid_files = [f for f in os.listdir(input_dir) if f.endswith((".xml", ".txt"))]
    print(f"Starte Inferenz-Extraktion für {len(valid_files)} Dateien im Ordner {input_dir}...")

    for filename in valid_files:
        filepath = os.path.join(input_dir, filename)
        text = extract_text(filepath)

        if not text:
            continue

        doc = nlp(text)

        current_w = Counter()
        current_p_tags = []
        current_m = Counter()
        current_length = 0
        chunk_index = 0

        for sent in doc.sents:

            if is_bible_quote(sent.text, bible_vectorizer, bible_matrix, threshold=0.45):
                continue

            valid_tokens = [t for t in sent if not t.is_punct and not t.is_space]
            if not valid_tokens:
                continue

            for t in valid_tokens:
                lemma = normalize_greek_token(t)
                if lemma in GREEK_FUNCTION_WORDS_LEMMATA:
                    current_w[lemma] += 1

                if t.morph:
                    current_m[str(t.morph)] += 1

                current_p_tags.append(t.pos_)

            current_length += len(valid_tokens)

            if current_length >= 1000:
                p_c = Counter(list(ngrams(current_p_tags, 3)))

                row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}

                for w in top_words:
                    row[f"LEMMA_{w}"] = current_w.get(w, 0)

                pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
                for p in top_pos:
                    row[f"POS_{p}"] = pos_dict.get(p, 0)

                for m in top_morph:
                    row[f"MORPH_{m}"] = current_m.get(m, 0)

                sample_records.append(row)

                current_w = Counter()
                current_p_tags = []
                current_m = Counter()
                current_length = 0
                chunk_index += 1

        # Letzten Rest-Chunk oder kurzes Gesamtdokument verarbeiten
        if current_length >= 500 or (chunk_index == 0 and current_length >= 100):
            if chunk_index == 0 and current_length < 500:
                print(
                    f"[Methodische Warnung] Text '{filename}' (Inferenz) ist mit {current_length} Tokens extrem kurz. Stilometrische Resultate sind hier statistisch instabil!")

            p_c = Counter(list(ngrams(current_p_tags, 3)))

            row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}
            for w in top_words:
                row[f"LEMMA_{w}"] = current_w.get(w, 0)
            pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
            for p in top_pos:
                row[f"POS_{p}"] = pos_dict.get(p, 0)
            for m in top_morph:
                row[f"MORPH_{m}"] = current_m.get(m, 0)

            sample_records.append(row)

    pd.DataFrame(sample_records).fillna(0).to_csv(output_csv, index=False)
    print(f"-> Inference Features formatiert und gespeichert in '{output_csv}'")


if __name__ == "__main__":
    INFERENCE_FOLDER = "data/inference/pseudo_corpus"
    OUTPUT_FILE = "inference_features.csv"
    VOCAB_FILE = "top_features_vocabulary.json"

    if not os.path.exists(INFERENCE_FOLDER):
        os.makedirs(INFERENCE_FOLDER)
        print(f"Ordner '{INFERENCE_FOLDER}' existierte nicht und wurde angelegt. Bitte Texte dort ablegen.")
    else:
        process_inference_corpus(INFERENCE_FOLDER, OUTPUT_FILE, VOCAB_FILE)