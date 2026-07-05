import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_publication_plots(csv_path, threshold=2.2736):
    print(f"Lade finale Daten aus {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Daten filtern (Wir zeigen nur die Top 5 Treffer und 5 klare Ablehnungen)
    # So wird das Diagramm nicht durch 30+ Balken überladen.
    df_sorted = df.sort_values(by='Mean_Distance')
    top_candidates = df_sorted.head(5)
    rejected_candidates = df_sorted.tail(5)
    
    # Verbinde die beiden Gruppen für einen starken Kontrast
    plot_df = pd.concat([top_candidates, rejected_candidates])
    
    # Bereinige die Dateinamen für eine schönere Beschriftung
    plot_df['Document_Clean'] = plot_df['Document'].apply(lambda x: str(x).replace('.xml', '').replace('.txt', ''))
    
    # 2. Farb-Logik basierend auf dem Threshold
    # Asterius-Treffer (unter Threshold) = Blau, Abgewiesen (über Threshold) = Grau
    plot_df['Color'] = np.where(plot_df['Mean_Distance'] <= threshold, '#1f77b4', '#d3d3d3')

    # 3. Canvas einrichten (Academical Style)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 4. Balkendiagramm zeichnen
    bars = ax.barh(plot_df['Document_Clean'], plot_df['Mean_Distance'], color=plot_df['Color'])
    
    # 5. Den kritischen Threshold einzeichnen
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Intra-Autor Threshold ({threshold:.2f})')
    
    # 6. Optischer Feinschliff
    ax.set_xlabel('Mittlere Euklidische Distanz zum Asterius-Zentroid', fontsize=12, fontweight='bold')
    ax.set_ylabel('Untersuchte Predigten (Pseudo-Korpus)', fontsize=12, fontweight='bold')
    ax.set_title('Autorschafts-Attribution: Trennschärfe des Siamese Networks', fontsize=14, fontweight='bold', pad=20)
    
    # Achsen invertieren, damit der geringste (beste) Wert ganz oben steht
    ax.invert_yaxis() 
    
    # Legende hinzufügen
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    
    # Werte direkt an die Balken schreiben für maximale Lesbarkeit
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.1 if width < 6 else width - 0.5
        color = 'black' if width < 6 else 'white'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', color=color, fontweight='bold')

    plt.tight_layout()
    
    # 7. Speichern in hoher Druckqualität (300 dpi)
    output_file = 'attribution_diverging_bar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Diagramm in hoher Qualität gespeichert unter: {output_file}")
    plt.show()

if __name__ == "__main__":
    # Stelle sicher, dass dies exakt der Dateiname aus dem vorherigen Schritt ist
    FINAL_CSV = "asterius_final_document_scores.csv" 
    
    # Der Threshold von 2.2736 stammt aus dem Inferenz-Skript
    generate_publication_plots(FINAL_CSV, threshold=2.2736)