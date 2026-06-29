import os
import re
import pandas as pd
import json
from collections import Counter
from nltk.util import ngrams
import spacy
from bs4 import BeautifulSoup

# 1. NLP-Modell laden
spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")

# 2. Globale Konstante (Muss im Skript vorhanden sein, sonst NameError bei f_lemmas)
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


def extract_text(filepath):
    """Liest Text und wendet strikten Unicode-Regex für Altgriechisch an."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = BeautifulSoup(f, 'lxml-xml').get_text(separator=' ') if filepath.endswith(".xml") else f.read()

    # Behält NUR Leerzeichen und Zeichen aus den griechischen/polytonischen Unicode-Blöcken
    clean_text = re.sub(r'[^\u0370-\u03FF\u1F00-\u1FFF\s]', ' ', raw_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text.strip()


def extract_sample_features(text_sample):
    """Extrahiert die philologischen Merkmale aus einem 1000-Wort-Sample."""
    doc = nlp(text_sample)

    lemmas = [t.lemma_ for t in doc if not t.is_punct and not t.is_space]
    f_lemmas = [l for l in lemmas if l in GREEK_FUNCTION_WORDS_LEMMATA]

    pos_tags = [t.pos_ for t in doc if not t.is_space]
    morphs = [str(t.morph) for t in doc if str(t.morph)]

    # Hier trat der NameError auf: Counter und ngrams sind nun sauber importiert
    return Counter(f_lemmas), Counter(list(ngrams(pos_tags, 3))), Counter(morphs)


def process_inference_corpus(input_dir, output_csv, vocab_json):
    """Erzwingt das Einpassen der unbekannten Texte in den trainierten Vektorraum."""
    if not os.path.exists(vocab_json):
        raise FileNotFoundError(f"Vokabular-Datei '{vocab_json}' fehlt. Führe erst extract_train_features.py aus!")

    with open(vocab_json, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    top_words = vocab['words']
    top_pos = vocab['pos']
    top_morph = vocab['morph']

    sample_records = []

    valid_files = [f for f in os.listdir(input_dir) if f.endswith((".xml", ".txt"))]
    print(f"Starte Inferenz-Extraktion für {len(valid_files)} Dateien im Ordner {input_dir}...")

    for filename in valid_files:
        filepath = os.path.join(input_dir, filename)
        words = extract_text(filepath).split()

        chunk_size = 1000
        for i in range(0, len(words), chunk_size):
            sample = words[i:i + chunk_size]
            if len(sample) != chunk_size:
                continue  # Verwirft Reste am Ende des Textes

            w_c, p_c, m_c = extract_sample_features(" ".join(sample))

            # Mapping auf das starre Vokabular des Trainings-Sets
            row = {"Auteur": "Pseudo", "Titre": f"{filename}_{i // chunk_size}"}

            for w in top_words:
                row[f"LEMMA_{w}"] = w_c.get(w, 0)

            pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
            for p in top_pos:
                row[f"POS_{p}"] = pos_dict.get(p, 0)

            for m in top_morph:
                row[f"MORPH_{m}"] = m_c.get(m, 0)

            sample_records.append(row)

    pd.DataFrame(sample_records).fillna(0).to_csv(output_csv, index=False)
    print(f"-> Inference Features formatiert und gespeichert in '{output_csv}'")


if __name__ == "__main__":
    INFERENCE_FOLDER = "data/inference/pseudo_corpus"
    OUTPUT_FILE = "inference_features.csv"
    VOCAB_FILE = "top_features_vocabulary.json"

    # Sicherheitsprüfung für den Ordner
    if not os.path.exists(INFERENCE_FOLDER):
        os.makedirs(INFERENCE_FOLDER)
        print(f"Ordner '{INFERENCE_FOLDER}' existierte nicht und wurde angelegt. Bitte Texte dort ablegen.")
    else:
        process_inference_corpus(INFERENCE_FOLDER, OUTPUT_FILE, VOCAB_FILE)