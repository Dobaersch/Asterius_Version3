import os
import re
import pandas as pd
import json
from collections import Counter
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
    "περί", "ἀντί", "πρό", "σύν", "ἄνευ", "ἕνεκα"
}


def clean_text(text):
    """
    Dynamische Textbereinigung: Nutzt den XML-Parser nur für tatsächliche XML-Daten,
    um reine .txt-Dokumente (wie die Asterius-Referenzen) nicht zu löschen.
    """
    if '<' in text and '>' in text:
        try:
            text = BeautifulSoup(text, "xml").get_text()
        except Exception:
            # Notfall-Fallback, falls das XML stark beschädigt ist
            text = re.sub(r'<[^>]+>', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_bible_vectorizer(bible_path="greek_bible.txt"):
    if not os.path.exists(bible_path):
        print(f"[Warnung] Bibel-Referenztext '{bible_path}' nicht gefunden. Zitat-Filterung ist inaktiv.")
        return None, None

    with open(bible_path, 'r', encoding='utf-8') as f:
        bible_text = f.read()

    # Reine Text-Bereinigung (ohne XML-Parser), da es sich um eine .txt-Datei handelt
    bible_text = re.sub(r'\s+', ' ', bible_text).strip()

    bible_sentences = re.split(r'[.·;]+', bible_text)
    bible_sentences = [s.strip() for s in bible_sentences if len(s.strip()) > 10]

    # Sicherheits-Check: Verhindert den 'empty vocabulary' Fehler
    if not bible_sentences:
        print(f"[Warnung] Die Datei '{bible_path}' enthält keine verwertbaren Sätze. Zitat-Filterung ist inaktiv.")
        return None, None

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    bible_tfidf = vectorizer.fit_transform(bible_sentences)
    return vectorizer, bible_tfidf


def extract_inference_features():
    INFERENCE_FOLDER = "data/inference/pseudo_corpus"
    vocab_json = "top_features_vocabulary.json"
    output_csv = "inference_features.csv"

    print("--- Starte Feature-Extraktion (Inferenzkorpus/Rolling Window) ---")

    if not os.path.exists(vocab_json):
        raise FileNotFoundError(
            f"[Kritischer Fehler] Vokabular '{vocab_json}' fehlt. Zuerst Trainingsskript ausführen.")

    with open(vocab_json, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        top_words = vocab['words']
        top_pos = vocab['pos']
        top_morph = vocab['morph']

    vectorizer, bible_tfidf = build_bible_vectorizer()

    def is_bible_quote(sentence_text, threshold=0.85):
        if not vectorizer or len(sentence_text) < 15: return False
        s_vec = vectorizer.transform([sentence_text])
        return cosine_similarity(s_vec, bible_tfidf).max() >= threshold

    sample_records = []

    for filename in os.listdir(INFERENCE_FOLDER):
        file_path = os.path.join(INFERENCE_FOLDER, filename)
        if not os.path.isfile(file_path): continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = clean_text(f.read())
        except Exception as e:
            print(f"Fehler bei {filename}: {e}")
            continue

        doc = nlp(raw_text)

        current_w = {}
        current_m = {}
        current_syntactic_trigrams = []

        current_length = 0
        chunk_index = 0

        for sent in doc.sents:
            if not is_bible_quote(sent.text):
                current_length += len(sent)
                for token in sent:
                    if token.lemma_ in GREEK_FUNCTION_WORDS_LEMMATA:
                        current_w[token.lemma_] = current_w.get(token.lemma_, 0) + 1

                    if token.morph:
                        morph_str = str(token.morph)
                        current_m[morph_str] = current_m.get(morph_str, 0) + 1

                    # Syntaktische POS-Trigramme extrahieren
                    children = sorted(list(token.children), key=lambda c: c.i)
                    if len(children) >= 2:
                        for i in range(len(children) - 1):
                            trigram = (children[i].pos_, token.pos_, children[i + 1].pos_)
                            current_syntactic_trigrams.append(trigram)

            # Rolling Window abspeichern bei >= 500 Tokens
            if current_length >= 500:
                p_c = Counter(current_syntactic_trigrams)

                row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}
                for w in top_words: row[f"LEMMA_{w}"] = current_w.get(w, 0)

                pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
                for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
                for m in top_morph: row[f"MORPH_{m}"] = current_m.get(m, 0)

                sample_records.append(row)

                # Reset für den nächsten Chunk
                current_w = {}
                current_m = {}
                current_syntactic_trigrams = []
                current_length = 0
                chunk_index += 1

        # Reste verarbeiten
        if current_length >= 500 or (chunk_index == 0 and current_length >= 100):
            if chunk_index == 0 and current_length < 500:
                print(
                    f"[Methodische Warnung] Text '{filename}' (Inferenz) ist mit {current_length} Tokens extrem kurz. Stilometrische Resultate sind hier statistisch instabil!")

            p_c = Counter(current_syntactic_trigrams)

            row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}
            for w in top_words: row[f"LEMMA_{w}"] = current_w.get(w, 0)

            pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
            for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
            for m in top_morph: row[f"MORPH_{m}"] = current_m.get(m, 0)

            sample_records.append(row)

    pd.DataFrame(sample_records).fillna(0).to_csv(output_csv, index=False)
    print(f"-> Inference Features formatiert und gespeichert in '{output_csv}'")


if __name__ == "__main__":
    extract_inference_features()