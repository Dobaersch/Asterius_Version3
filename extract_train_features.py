import os
import pandas as pd
import json
from collections import Counter
from nltk.util import ngrams
import spacy
from bs4 import BeautifulSoup

spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")

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

def extract_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f, 'lxml-xml').get_text(separator=' ') if filepath.endswith(".xml") else f.read()

def extract_sample_features(text_sample):
    doc = nlp(text_sample)
    lemmas = [t.lemma_ for t in doc if not t.is_punct and not t.is_space]
    f_lemmas = [l for l in lemmas if l in GREEK_FUNCTION_WORDS_LEMMATA]
    pos_tags = [t.pos_ for t in doc if not t.is_space]
    morphs = [str(t.morph) for t in doc if str(t.morph)]
    
    return Counter(f_lemmas), Counter(list(ngrams(pos_tags, 3))), Counter(morphs)

def process_training_corpus(input_dirs, output_csv, vocab_json):
    sample_records = []
    global_counts = {'words': Counter(), 'pos': Counter(), 'morph': Counter()}
    
    # Durchsuche Asterius, Chrysostomos, Severian etc.
    for folder in input_dirs:
        if not os.path.exists(folder): continue
        for filename in [f for f in os.listdir(folder) if f.endswith((".xml", ".txt"))]:
            filepath = os.path.join(folder, filename)
            author = os.path.basename(folder) # Ordnername = Autor
            words = extract_text(filepath).split()
            
            chunk_size = 1000
            for i in range(0, len(words), chunk_size):
                sample = words[i:i+chunk_size]
                if len(sample) != chunk_size: continue
                
                w_c, p_c, m_c = extract_sample_features(" ".join(sample))
                global_counts['words'].update(w_c)
                global_counts['pos'].update(p_c)
                global_counts['morph'].update(m_c)
                
                sample_records.append({
                    "author": author, "title": f"{filename}_{i//chunk_size}",
                    "w": w_c, "p": p_c, "m": m_c
                })

    # Ermittle Top Features (z.B. Top 100 Lemmata, Top 100 POS, Top 100 Morph)
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
        for w in top_words: row[f"LEMMA_{w}"] = r["w"].get(w, 0)
        # Format POS-Tuple to String for dict lookup
        pos_dict = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in r["p"].items()}
        for p in top_pos: row[f"POS_{p}"] = pos_dict.get(p, 0)
        for m in top_morph: row[f"MORPH_{m}"] = r["m"].get(m, 0)
        all_features.append(row)
        
    pd.DataFrame(all_features).fillna(0).to_csv(output_csv, index=False)
    print(f"Training Features gespeichert in {output_csv}")

if __name__ == "__main__":
    train_dirs = ["data/train/asterius", "data/train/hard_negatives", "data/train/easy_negatives"]
    process_training_corpus(train_dirs, "train_features.csv", "top_features_vocabulary.json")