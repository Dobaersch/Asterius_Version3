import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_publication_plots(csv_path, threshold=2.2736):
    print(f"Lade finale Daten aus {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Fehler] Die Datei '{csv_path}' wurde nicht gefunden. Bitte zuerst aggregate_results.py ausführen.")
        return

    # 1. Daten filtern (Wir zeigen nur die Top 5 Treffer und 5 klare Ablehnungen)
    # KORRIGIERT: Neuer Spaltenname 'Mean_Distance_to_Asterius'
    df_sorted = df.sort_values(by='Mean_Distance_to_Asterius')
    top_candidates = df_sorted.head(5)
    rejected_candidates = df_sorted.tail(5)

    # Verbinde die beiden Gruppen für einen starken Kontrast
    plot_df = pd.concat([top_candidates, rejected_candidates])

    # Bereinige die Dateinamen für eine schönere Beschriftung
    # KORRIGIERT: Zugrunde liegende Spalte heißt nun 'Document'
    plot_df['Document_Clean'] = plot_df['Document'].apply(lambda x: str(x).replace('.xml', '').replace('.txt', ''))

    # 2. Farb-Logik basierend auf dem Threshold
    # Asterius-Treffer (unter Threshold) = Blau, Abgewiesen (über Threshold) = Grau
    plot_df['Color'] = np.where(plot_df['Mean_Distance_to_Asterius'] <= threshold, '#1f77b4', '#d3d3d3')

    # 3. Canvas einrichten (Academical Style)
    fig, ax = plt.subplots(figsize=(10, 8))

    # 4. Balkendiagramm zeichnen
    bars = ax.barh(plot_df['Document_Clean'], plot_df['Mean_Distance_to_Asterius'], color=plot_df['Color'])

    # 5. Demarkationslinie (Threshold)
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Demarkationslinie (Threshold: {threshold:.2f})')

    # 6. Styling
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
        label_x_pos = width + 0.1 if width < (threshold * 1.5) else width - 0.5
        color = 'black' if width < (threshold * 1.5) else 'white'
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                va='center', color=color, fontweight='bold')

    plt.tight_layout()

    # 7. Speichern in hoher Druckqualität (300 dpi)
    output_file = 'attribution_diverging_bar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot erfolgreich gespeichert: {output_file}")


if __name__ == "__main__":
    # KORRIGIERT: Lade die aggregierte Dokumenten-Tabelle anstelle der Chunk-Tabelle
    FINAL_CSV = "asterius_final_document_scores.csv"

    # HINWEIS: Trage hier deinen aktuellen dynamischen ML-Schwellenwert ein!
    # (Dieser wurde dir im Terminal beim Durchlauf von infer_pseudo_corpus.py und aggregate_results.py ausgegeben)
    AKTUELLER_THRESHOLD = 2.2736

    generate_publication_plots(FINAL_CSV, threshold=AKTUELLER_THRESHOLD)