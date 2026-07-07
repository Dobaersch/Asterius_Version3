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

# Unterdrücke Warnungen für eine saubere Konsole
warnings.simplefilter(action='ignore', category=FutureWarning)

# NLP-Pipeline laden
spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")
nlp.max_length = 3000000  # Puffer massiv erhöht für sehr lange Predigten

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

GREEK_FUNCTION_WORDS_LEMMATA = {
    "καί", "δέ", "τε", "ἀλλά", "ἤ", "γάρ", "οὖν", "ἄρα", "διό",
    "ἵνα", "ὅπως", "ὡς", "ὥστε", "ὅτι", "εἰ", "ἐάν", "ἐπεί", "ἐπειδή",
    "οὔτε", "μήτε", "οὐδέ", "μηδέ", "πλήν", "ἐν", "εἰς", "ἐκ", "ἐξ",
    "πρός", "ἐπί", "διά", "κατά", "μετά", "παρά", "ἀπό", "ὑπέρ", "ὑπό",
    "περί", "ἀντί", "πρό", "σύν", "ἄνευ", "ἕνεκα"
}

def read_file_safely(file_path):
    """
    Versucht verschiedene Encodings. Altgriechische Textkorpora 
    weisen häufig Encoding-Inkonsistenzen auf.
    """
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
    und entfernt englischsprachige Metadaten (teiHeader).
    """
    if filename.lower().endswith('.xml'):
        # 1. Entferne eventuelle XML-Deklarationen, die den Wrapper stören würden
        text = re.sub(r'<\?xml.*?\?>', '', text).strip()

        # 2. Künstlicher Root-Knoten (<document>) repariert fragmentarische
        # XML-Dateien (wie Basilius), damit der Parser nicht abstürzt.
        wrapped_text = f"<document>{text}</document>"

        try:
            soup = BeautifulSoup(wrapped_text, "xml")

            # 3. Projekt-Optimierung: TEI-Header (Metadaten) restlos löschen,
            # da diese sonst die syntaktische Trigramm-Statistik verfälschen!
            for header in soup.find_all('teiHeader'):
                header.decompose()

            text = soup.get_text(separator=' ')
        except Exception:
            # Fallback
            text = re.sub(r'<[^>]+>', ' ', text)
    else:
        # Bei .txt Dateien entfernen wir nur die eckigen Klammern (Konjekturen),
        # damit rekonstruierte Wörter (z.B. <θεὸς>) philologisch erhalten bleiben.
        text = text.replace('<', '').replace('>', '')

    # Bereinige doppelte Leerzeichen und Zeilenumbrüche
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_bible_vectorizer(bible_path="greek_bible.txt"):
    if not os.path.exists(bible_path):
        print(f"[Warnung] Bibel-Referenztext '{bible_path}' nicht gefunden. Zitat-Filterung inaktiv.")
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

def extract_features():
    TRAIN_FOLDER = "data/train"
    vocab_json = "top_features_vocabulary.json"
    output_csv = "train_features.csv"

    print("--- Starte Feature-Extraktion (Trainingskorpus) ---")
    
    vectorizer, bible_tfidf = build_bible_vectorizer()

    def is_bible_quote(sentence_text, threshold=0.85):
        if not vectorizer or len(sentence_text) < 15:
            return False
        s_vec = vectorizer.transform([sentence_text])
        sim = cosine_similarity(s_vec, bible_tfidf).max()
        return sim >= threshold

    global_counts = {'words': Counter(), 'pos': Counter(), 'morph': Counter()}
    sample_records = []

    for author in os.listdir(TRAIN_FOLDER):
        author_path = os.path.join(TRAIN_FOLDER, author)
        if not os.path.isdir(author_path): continue

        for filename in os.listdir(author_path):
            file_path = os.path.join(author_path, filename)
            
            # Systemdateien und Verzeichnisse zwingend ignorieren
            if not os.path.isfile(file_path) or filename.startswith('.'):
                continue
                
            try:
                raw_text = read_file_safely(file_path)
                raw_text = clean_text(raw_text, filename)
                
                if len(raw_text) < 50:
                    print(f"[Übersprungen] '{filename}' enthält zu wenig Text.")
                    continue
                    
                doc = nlp(raw_text)
            except Exception as e:
                # Hier lag der Fehler: Silent Fails wurden abgefangen und ignoriert
                print(f"[Kritischer Fehler] Datei '{filename}' übersprungen: {e}")
                continue
            
            current_w = Counter()
            current_m = Counter()
            current_syntactic_trigrams = []

            for sent in doc.sents:
                if not is_bible_quote(sent.text):
                    for token in sent:
                        if token.lemma_ in GREEK_FUNCTION_WORDS_LEMMATA:
                            current_w[token.lemma_] += 1
                            global_counts['words'][token.lemma_] += 1

                        if token.morph:
                            morph_str = str(token.morph)
                            current_m[morph_str] += 1
                            global_counts['morph'][morph_str] += 1

                        children = sorted(list(token.children), key=lambda c: c.i)
                        if len(children) >= 2:
                            for i in range(len(children) - 1):
                                trigram = (children[i].pos_, token.pos_, children[i+1].pos_)
                                current_syntactic_trigrams.append(trigram)
                                global_counts['pos'][trigram] += 1

            p_c = Counter(current_syntactic_trigrams)
            
            sample_records.append({
                "author": author,
                "title": filename,
                "w": dict(current_w),
                "p": dict(p_c),
                "m": dict(current_m)
            })

    if not sample_records:
        print("[Fehler] Es konnten keine Features extrahiert werden. Trainingskorpus leer?")
        return

    top_words = [w for w, _ in global_counts['words'].most_common(100)]
    top_pos = [f"{p[0]}_{p[1]}_{p[2]}" for p, _ in global_counts['pos'].most_common(100)]
    top_morph = [m for m, _ in global_counts['morph'].most_common(100)]

    with open(vocab_json, 'w', encoding='utf-8') as f:
        json.dump({'words': top_words, 'pos': top_pos, 'morph': top_morph}, f)

    all_features = []
    for r in sample_records:
        row = {"Auteur": r["author"], "Titre": r["title"]}
        for w in top_words: row[f"LEMMA_{w}"] = r["w"].get(w, 0)
        
        pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in r["p"].items()}
        for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
        
        for m in top_morph: row[f"MORPH_{m}"] = r["m"].get(m, 0)
        all_features.append(row)

    df_train = pd.DataFrame(all_features).fillna(0)
    df_train.to_csv(output_csv, index=False)
    print(f"-> Train Features gespeichert in '{output_csv}'")

if __name__ == "__main__":
    extract_features()