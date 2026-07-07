import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# --- KONFIGURATION ---
# Passe die Pfade an, falls deine Ordnerstruktur abweicht.
TRAIN_DIR_ASTERIUS = "data/train/asterius"
INFER_DIR_PSEUDO = "data/inference/pseudo_corpus"
OUTPUT_CSV = "baseline_evaluation_results.csv"
MAX_FEATURES = 1000 # Wir nutzen die 1000 häufigsten Zeichen-Trigramme

def clean_text(text):
    """
    Entfernt XML/HTML-Tags, da das Korpus teilweise aus .xml-Dateien besteht,
    sowie überflüssige Whitespaces. Im Altgriechischen belassen wir die Akzente 
    für die Baseline zunächst intakt, da sie syntaktische Informationen (z.B. Kasus) 
    mittragen können.
    """
    text = re.sub(r'<[^>]+>', ' ', text)  # XML-Tags entfernen
    text = re.sub(r'\s+', ' ', text)      # Mehrfache Leerzeichen reduzieren
    return text.strip()

def load_texts_from_directory(directory):
    """
    Lädt alle .txt und .xml Dateien aus einem angegebenen Verzeichnis.
    Gibt zwei Listen zurück: Dateinamen und die zugehörigen Rohtexte.
    """
    file_paths = glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.xml"))
    
    filenames = []
    texts = []
    
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
                cleaned = clean_text(raw_text)
                if len(cleaned) > 100: # Überspringe leere oder fehlerhafte Dateien
                    texts.append(cleaned)
                    filenames.append(os.path.basename(path))
        except Exception as e:
            print(f"Fehler beim Laden von {path}: {e}")
            
    return filenames, texts

def main():
    print("--- Starte Baseline-Evaluation (Cosine Delta / Char-Trigrams) ---")
    
    # 1. Daten laden
    print("[1] Lade Referenztexte (Asterius)...")
    ast_filenames, ast_texts = load_texts_from_directory(TRAIN_DIR_ASTERIUS)
    if not ast_texts:
        raise ValueError(f"Keine Asterius-Texte in {TRAIN_DIR_ASTERIUS} gefunden.")
        
    print("[2] Lade Inferenztexte (Pseudo-Chrysostomos)...")
    pc_filenames, pc_texts = load_texts_from_directory(INFER_DIR_PSEUDO)
    if not pc_texts:
        raise ValueError(f"Keine Pseudo-Chrysostomos-Texte in {INFER_DIR_PSEUDO} gefunden.")

    # 2. Vektorisierung (Feature Extraction)
    print(f"[3] Extrahiere die {MAX_FEATURES} häufigsten Zeichen-Trigramme (skip=0)...")
    # Wir trainieren den Vectorizer auf dem gesamten Korpus, um einen gemeinsamen Vektorraum aufzuspannen
    all_texts = ast_texts + pc_texts
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=MAX_FEATURES)
    X_all = vectorizer.fit_transform(all_texts).toarray()
    
    # Matrizen wieder trennen
    num_ast = len(ast_texts)
    X_ast = X_all[:num_ast]
    X_pc = X_all[num_ast:]
    
    # 3. Profilbildung & Distanzmessung
    print("[4] Berechne Asterius-Centroid und Cosinus-Distanzen...")
    # Das "Zentrum" des Asterius-Stils ist der Durchschnitt aller seiner Feature-Vektoren
    asterius_centroid = np.mean(X_ast, axis=0).reshape(1, -1)
    
    # Berechne die Distanz jedes PC-Textes zum Asterius-Centroid
    distances = cosine_distances(X_pc, asterius_centroid).flatten()
    
    # 4. Ergebnisse aggregieren und speichern
    results = []
    for filename, dist in zip(pc_filenames, distances):
        results.append({
            "Dokument": filename,
            "Baseline_Cosine_Distance": round(dist, 4)
        })
        
    df_results = pd.DataFrame(results)
    
    # Sortieren: Die Texte mit der geringsten Distanz stehen oben (höchste Asterius-Wahrscheinlichkeit)
    df_results = df_results.sort_values(by="Baseline_Cosine_Distance")
    
    df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[5] Erfolgreich abgeschlossen. Ergebnisse gespeichert in: {OUTPUT_CSV}")
    print("\n--- Top 5 Kandidaten (Geringste Distanz zum Asterius-Stil) ---")
    print(df_results.head(5).to_string(index=False))

if __name__ == "__main__":
    main()