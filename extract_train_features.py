import os
import re
import pandas as pd
import json
from collections import Counter
from nltk.util import ngrams
import spacy
from bs4 import BeautifulSoup

# NLP-Pipeline laden und Sentencizer sicherstellen
spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")
nlp.max_length = 2000000

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Liste der Funktionswörter
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

# Fallback-Matrix für Elisionen und phonetische Angleichungen
ELISION_MAP = {
    "ἀλλ'": "ἀλλά", "δι'": "διά", "κατ'": "κατά", "μεθ'": "μετά",
    "μετ'": "μετά", "παρ'": "παρά", "ἐπ'": "ἐπί", "ἐφ'": "ἐπί",
    "ὑπ'": "ὑπό", "ὑφ'": "ὑπό", "ἀπ'": "ἀπό", "ἀφ'": "ἀπό",
    "ἀντ'": "ἀντί", "ἀνθ'": "ἀντί", "οὐκ": "οὐ", "οὐχ": "οὐ",
    "τ'": "τε", "δ'": "δέ", "γ'": "γε", "μ'": "με"
}


def extract_text(filepath):
    """Liest Text ein und bereinigt ihn unter Beibehaltung der Interpunktion für NLP."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = BeautifulSoup(f, 'lxml-xml').get_text(separator=' ') if filepath.endswith(".xml") else f.read()

    # REGEX-FILTER: Behält griechische Buchstaben UND notwendige Interpunktion (.,;··!?) für die Syntaxanalyse.
    clean_text = re.sub(r'[^\u0370-\u03FF\u1F00-\u1FFF\s\.,;··!\?]', ' ', raw_text)

    # Mehrfache Leerzeichen durch ein einziges ersetzen
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text.strip()


def normalize_greek_token(token):
    """Sichert die Lemmatisierung gegen Elisionen und Krasis ab."""
    lemma = token.lemma_.lower() if token.lemma_ else token.text.lower()
    text_lower = token.text.lower()

    if text_lower in ELISION_MAP:
        return ELISION_MAP[text_lower]

    return lemma


def process_training_corpus(input_dirs, output_csv, vocab_json):
    sample_records = []
    global_counts = {'words': Counter(), 'pos': Counter(), 'morph': Counter()}

    # Durchsuche das definierte Trainingskorpus
    for folder in input_dirs:
        if not os.path.exists(folder):
            print(f"[Warnung] Verzeichnis nicht gefunden: {folder}")
            continue

        author = os.path.basename(folder)
        for filename in [f for f in os.listdir(folder) if f.endswith((".xml", ".txt"))]:
            filepath = os.path.join(folder, filename)
            text = extract_text(filepath)

            if not text:
                continue

            # NLP Parsing des gesamten Dokuments (Sentencizer ist aktiv)
            doc = nlp(text)

            current_w = Counter()
            current_p_tags = []
            current_m = Counter()
            current_length = 0
            chunk_index = 0

            # Iteration auf syntaktischer Ebene (Satz für Satz)
            for sent in doc.sents:
                valid_tokens = [t for t in sent if not t.is_punct and not t.is_space]
                if not valid_tokens:
                    continue

                # Token-Eigenschaften extrahieren
                for t in valid_tokens:
                    lemma = normalize_greek_token(t)
                    if lemma in GREEK_FUNCTION_WORDS_LEMMATA:
                        current_w[lemma] += 1

                    if t.morph:
                        current_m[str(t.morph)] += 1

                    current_p_tags.append(t.pos_)

                current_length += len(valid_tokens)

                # Chunk schließen, sobald ~1000 Wörter erreicht sind (Satzende gewahrt)
                if current_length >= 1000:
                    # POS-Trigramme über den gesamten, syntaktisch intakten Chunk berechnen
                    p_c = Counter(list(ngrams(current_p_tags, 3)))

                    global_counts['words'].update(current_w)
                    global_counts['pos'].update(p_c)
                    global_counts['morph'].update(current_m)

                    sample_records.append({
                        "author": author,
                        "title": f"{filename}_{chunk_index}",
                        "w": current_w,
                        "p": p_c,
                        "m": current_m
                    })

                    # Reset für den nächsten Chunk
                    current_w = Counter()
                    current_p_tags = []
                    current_m = Counter()
                    current_length = 0
                    chunk_index += 1

            # Letzten Rest-Chunk verarbeiten, falls er statistisch belastbar ist (> 500 Wörter)
            if current_length >= 500:
                p_c = Counter(list(ngrams(current_p_tags, 3)))
                global_counts['words'].update(current_w)
                global_counts['pos'].update(p_c)
                global_counts['morph'].update(current_m)

                sample_records.append({
                    "author": author,
                    "title": f"{filename}_{chunk_index}",
                    "w": current_w,
                    "p": p_c,
                    "m": current_m
                })

    if not sample_records:
        print("[Fehler] Es konnten keine Features extrahiert werden. Trainingskorpus leer?")
        return

    # Ermittle Top Features (Top 100 Lemmata, Top 100 POS, Top 100 Morph)
    top_words = [w for w, _ in global_counts['words'].most_common(100)]
    top_pos = [f"{p[0]}_{p[1]}_{p[2]}" for p, _ in global_counts['pos'].most_common(100)]
    top_morph = [m for m, _ in global_counts['morph'].most_common(100)]

    # SPEICHERN des Vokabulars für das Inferenz-Skript
    with open(vocab_json, 'w', encoding='utf-8') as f:
        json.dump({'words': top_words, 'pos': top_pos, 'morph': top_morph}, f)

    # Matrix bauen
    all_features = []
    for r in sample_records:
        row = {"Auteur": r["author"], "Titre": r["title"]}

        for w in top_words:
            row[f"LEMMA_{w}"] = r["w"].get(w, 0)

        # Format POS-Tuple zu String für die Wörterbuchabfrage
        pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in r["p"].items()}
        for p in top_pos:
            row[f"POS_{p}"] = pos_dict.get(p, 0)

        for m in top_morph:
            row[f"MORPH_{m}"] = r["m"].get(m, 0)

        all_features.append(row)

    # DataFrame exportieren
    df = pd.DataFrame(all_features).fillna(0)
    df.to_csv(output_csv, index=False)
    print(f"Training Features erfolgreich extrahiert und gespeichert in: {output_csv}")


if __name__ == "__main__":
    train_dirs = [
        "data/train/asterius",
        "data/train/chrysostomos",
        "data/train/severian",
        "data/train/gregor_nyssa",
        "data/train/gregor_nazianz",
        "data/train/basilius"
    ]
    process_training_corpus(train_dirs, "train_features.csv", "top_features_vocabulary.json")