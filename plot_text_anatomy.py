import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def analyze_text_anatomy(csv_path, target_document, threshold=2.2736):
    print(f"Analysiere interne Struktur für: {target_document}")
    
    # 1. Daten laden
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Datei {csv_path} nicht gefunden.")
    df = pd.read_csv(csv_path)
    
    # 2. Relevante Predigt filtern
    # Wir suchen nach allen Zeilen, die den Zielnamen (ohne Segment-Endung) enthalten
    doc_df = df[df['Pseudo_Text_Titre'].str.contains(target_document, case=False, regex=False)].copy()
    
    if doc_df.empty:
        raise ValueError(f"Keine Segmente für '{target_document}' gefunden. Bitte Dateinamen prüfen.")
        
    # 3. Textsegmente chronologisch ordnen
    # Extrahiert die Zahl am Ende des Strings (z.B. aus 'CPG4618.xml_2' wird 2)
    def extract_segment_id(title):
        match = re.search(r'_(\d+)$', str(title))
        return int(match.group(1)) if match else 0
        
    doc_df['Segment_ID'] = doc_df['Pseudo_Text_Titre'].apply(extract_segment_id)
    doc_df = doc_df.sort_values(by='Segment_ID')
    
    # 4. Canvas einrichten (Präsentations-Stil)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 5. Verlaufslinie (Rolling Window) zeichnen
    ax.plot(doc_df['Segment_ID'], doc_df['Distance_to_Centroid'], 
            marker='o', markersize=10, linewidth=3, color='#2c3e50', label='Distanzverlauf des Textes')
    
    # 6. Referenzbereiche (Zonen) definieren
    # Rote harte Linie für den Threshold
    ax.axhline(y=threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Intra-Autor Threshold ({threshold:.2f})')
    
    # Grüne Zone: Core-Asterius
    ax.axhspan(ymin=0, ymax=threshold, color='#2ecc71', alpha=0.15, label='Asterius-Kernbereich')
    
    # Gelbe Zone: Kompilations-Grauzone (Threshold + 25%)
    grauzone_max = threshold * 1.25
    ax.axhspan(ymin=threshold, ymax=grauzone_max, color='#f1c40f', alpha=0.15, label='Grauzone / Rauschen')
    
    # 7. Optischer Feinschliff für Folien
    ax.set_title(f'Stilometrische Anatomie: {target_document}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Textverlauf (1000-Wort-Segmente)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Euklidische Distanz zum Asterius-Centroid', fontsize=12, fontweight='bold')
    
    # X-Achse zwingend auf ganze Zahlen setzen (es gibt kein Segment 1.5)
    ax.set_xticks(doc_df['Segment_ID'])
    ax.set_xticklabels([f"Seg. {i}" for i in doc_df['Segment_ID']])
    
    # Y-Achse sinnvoll skalieren (minimaler Puffer oben/unten)
    max_dist = max(doc_df['Distance_to_Centroid'].max(), grauzone_max)
    ax.set_ylim(0, max_dist + 1.0)
    
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    output_filename = f"anatomy_{target_document.split('.')[0]}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"[OK] Visualisierung gespeichert als {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Pfad zur Datei VOR der Aggregation
    INFERENCE_CSV = "asterius_inference_results.csv" 
    
    # Trage hier den exakten Basisnamen eines Textes ein, der aus mehreren Segmenten besteht.
    # Beispiel: Wenn du den Verlauf von CPG4618 sehen willst:
    ZIELDOKUMENT = "AsteriusIgnotus_InPsalmum18"
    
    analyze_text_anatomy(INFERENCE_CSV, ZIELDOKUMENT)