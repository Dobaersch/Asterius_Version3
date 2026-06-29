import os
import pandas as pd
import json
from collections import Counter
from nltk.util import ngrams
import spacy
from bs4 import BeautifulSoup

spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")

# Die Funktion extract_sample_features und extract_text aus Skript A kopieren
# ... (Code für extract_text und extract_sample_features hier identisch einfügen) ...

def process_inference_corpus(input_dir, output_csv, vocab_json):
    if not os.path.exists(vocab_json):
        raise FileNotFoundError("Vokabular-Datei nicht gefunden. Führe erst extract_train_features.py aus!")
        
    with open(vocab_json, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    top_words = vocab['words']
    top_pos = vocab['pos']
    top_morph = vocab['morph']

    sample_records = []
    
    for filename in [f for f in os.listdir(input_dir) if f.endswith((".xml", ".txt"))]:
        filepath = os.path.join(input_dir, filename)
        words = extract_text(filepath).split()
        
        chunk_size = 1000
        for i in range(0, len(words), chunk_size):
            sample = words[i:i+chunk_size]
            if len(sample) != chunk_size: continue
            
            w_c, p_c, m_c = extract_sample_features(" ".join(sample))
            
            # Direktes Mappen auf das Vorgegebene Vokabular
            row = {"Auteur": "Pseudo", "Titre": f"{filename}_{i//chunk_size}"}
            for w in top_words: row[f"LEMMA_{w}"] = w_c.get(w, 0)
            
            pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in p_c.items()}
            for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
            
            for m in top_morph: row[f"MORPH_{m}"] = m_c.get(m, 0)
            
            sample_records.append(row)

    pd.DataFrame(sample_records).fillna(0).to_csv(output_csv, index=False)
    print(f"Inference Features formatiert und gespeichert in {output_csv}")

if __name__ == "__main__":
    process_inference_corpus("data/inference/pseudo_corpus", "inference_features.csv", "top_features_vocabulary.json")