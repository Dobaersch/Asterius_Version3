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

spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")
nlp.max_length = 3000000

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

def read_file_safely(file_path):
    """Sicheres Einlesen mit Encoding-Fallback"""
    encodings = ['utf-8-sig', 'utf-8', 'iso-8859-7', 'windows-1253', 'latin-1']
    last_error = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise ValueError(f"Konnte nicht decodiert werden. Letzter Fehler: {last_error}")


def clean_text(text, filename):
    """
    Dynamisches Parsing für heterogene Korpora: Repariert XML-Fragmente
    und entfernt Metadaten (teiHeader).
    """
    if filename.lower().endswith('.xml'):
        # 1. Entferne eventuelle XML-Deklarationen, die den Wrapper stören würden
        text = re.sub(r'<\?xml.*?\?>', '', text).strip()

        # 2. Künstlicher Root-Knoten (<document>) repariert fragmentarische
        # XML-Dateien (wie Basilius), damit der Parser nicht abstürzt.
        wrapped_text = f"<document>{text}</document>"

        try:
            soup = BeautifulSoup(wrapped_text, "xml")

            # TEI-Header (Metadaten) restlos löschen,
            # da diese sonst die syntaktische Trigramm-Statistik verfälschen!
            for header in soup.find_all('teiHeader'):
                header.decompose()

            text = soup.get_text(separator=' ')
        except Exception:
            # Fallback
            text = re.sub(r'<[^>]+>', ' ', text)
    else:
        # Entfernen von eckigen Klammern (Konjekturen),
        # damit rekonstruierte Wörter (z.B. <θεὸς>) erhalten bleiben.
        text = text.replace('<', '').replace('>', '')

    # Bereinige doppelte Leerzeichen und Zeilenumbrüche
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_bible_vectorizer(bible_path="greek_bible.txt"):
    if not os.path.exists(bible_path):
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
    INFERENCE_FOLDER = "data/inference/pseudo_corpus"
    vocab_json = "top_features_vocabulary.json"
    output_csv = "inference_features.csv"

    print("--- Starte Feature-Extraktion (Inferenzkorpus/Rolling Window) ---")

    if not os.path.exists(vocab_json):
        raise FileNotFoundError(
            f"[Kritischer Fehler] Vokabular '{vocab_json}' fehlt. Zuerst Trainingsskript ausführen.")

    # Vokabular aus der Trainingsphase laden
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

        # Ignoriere Unterordner oder Systemdateien
        if not os.path.isfile(file_path) or filename.startswith('.'):
            continue

        try:
            raw_text = read_file_safely(file_path)
            raw_text = clean_text(raw_text, filename)

            if len(raw_text) < 50:
                print(f"[Übersprungen] '{filename}' enthält zu wenig verwertbaren Text.")
                continue

            doc = nlp(raw_text)
        except Exception as e:
            print(f"[Kritischer Fehler] Datei '{filename}' übersprungen: {e}")
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
                    # Dynamische Erfassung aller alphabetischen Tokens
                    if token.is_alpha:
                        lemma = token.lemma_.lower()
                        current_w[lemma] = current_w.get(lemma, 0) + 1

                    if token.morph:
                        morph_str = str(token.morph)
                        current_m[morph_str] = current_m.get(morph_str, 0) + 1

                    # Syntaktische POS-Trigramme
                    children = sorted(list(token.children), key=lambda c: c.i)
                    if len(children) >= 2:
                        for i in range(len(children) - 1):
                            trigram = (children[i].pos_, token.pos_, children[i + 1].pos_)
                            current_syntactic_trigrams.append(trigram)

            # Rolling Window: Angehoben auf >= 1000 Tokens
            if current_length >= 1000:
                p_c = Counter(current_syntactic_trigrams)

                row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}

                # Zuweisung basierend auf dem geladenen JSON-Vokabular
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

        # Restlichen Text verarbeiten (letzter Chunk)
        if current_length >= 1000 or (chunk_index == 0 and current_length >= 250):
            if chunk_index == 0 and current_length < 1000:
                print(
                    f"[Methodische Warnung] Text '{filename}' ist mit {current_length} Tokens sehr kurz (Resultate statistisch instabil).")

            p_c = Counter(current_syntactic_trigrams)

            row = {"Auteur": "Pseudo", "Titre": f"{filename}_{chunk_index}"}
            for w in top_words: row[f"LEMMA_{w}"] = current_w.get(w, 0)

            pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
            for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
            for m in top_morph: row[f"MORPH_{m}"] = current_m.get(m, 0)

            sample_records.append(row)

    df_infer = pd.DataFrame(sample_records).fillna(0)
    df_infer.to_csv(output_csv, index=False)
    print(f"-> Inference Features formatiert und gespeichert in '{output_csv}'")


if __name__ == "__main__":
    extract_inference_features()