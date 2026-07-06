import pandas as pd
import re


def aggregate_and_evaluate():
    # 1. Daten laden
    try:
        df = pd.read_csv("asterius_inference_results.csv")
    except FileNotFoundError:
        print("[Fehler] Die Datei 'asterius_inference_results.csv' wurde nicht gefunden.")
        return

    # Nur Vergleiche mit dem Zielautor "Asterius" filtern
    df_asterius = df[df['ComparatorClass'].str.lower() == 'asterius'].copy()

    # 2. Document-Level Aggregation
    # Entfernt das Suffix "_sample_X" aus "AsteriusIgnotus_InPsalmum9_sample_1"
    def extract_doc_name(sample_label):
        return re.sub(r'_sample_\d+$', '', str(sample_label))

    df_asterius['Document'] = df_asterius['ComparedLabel'].apply(extract_doc_name)

    # Distanz-Mittelwert pro Dokument zum Asterius-Centroid berechnen
    doc_distances = df_asterius.groupby('Document')['Distance'].mean().reset_index()
    doc_distances.columns = ['Document', 'Mean_Distance_to_Asterius']

    # 3. Cluster Separation: "Asterius Ignotus" vs. "Restliches Pseudo-Korpus"
    # Die Dateinamen im Ignotus-Korpus beginnen mit "AsteriusIgnotus"
    ignotus_mask = doc_distances['Document'].str.contains('AsteriusIgnotus', case=False, na=False)

    df_ignotus = doc_distances[ignotus_mask]
    df_other = doc_distances[~ignotus_mask]

    # --- BEWEISSCHRITT 1: Intra-Cluster-Kohärenz (Einheitlichkeit) ---
    # Berechnung der Varianz und Standardabweichung
    var_ignotus = df_ignotus['Mean_Distance_to_Asterius'].var(ddof=0)
    std_ignotus = df_ignotus['Mean_Distance_to_Asterius'].std(ddof=0)

    var_other = df_other['Mean_Distance_to_Asterius'].var(ddof=0)
    std_other = df_other['Mean_Distance_to_Asterius'].std(ddof=0)

    # --- BEWEISSCHRITT 2: Inter-Cluster-Distanz (Nähe zu Asterius) ---
    mean_ignotus = df_ignotus['Mean_Distance_to_Asterius'].mean()
    mean_other = df_other['Mean_Distance_to_Asterius'].mean()

    # HIER ANPASSEN: Trage hier den Schwellenwert ein, den dein Training für Asterius ermittelt hat
    ASTERIUS_THRESHOLD = 1.15

    # --- REPORT GENERIERUNG ---
    print("=" * 70)
    print(" MAKRO-ANALYSE: ASTERIUS IGNOTUS CLUSTER VS. PSEUDO-KORPUS")
    print("=" * 70)
    print(f"Anzahl evaluierter 'Asterius Ignotus' Dokumente: {len(df_ignotus)}")
    print(f"Anzahl restlicher Pseudo-Dokumente:              {len(df_other)}\n")

    print("--- TEIL 1: INTRA-CLUSTER-KOHÄRENZ (Kinzigs Einheitlichkeit) ---")
    print(f"Varianz Asterius Ignotus:      {var_ignotus:.4f} (StdAbw: {std_ignotus:.4f})")
    print(f"Varianz Restliches Korpus:     {var_other:.4f} (StdAbw: {std_other:.4f})")

    if var_ignotus < var_other:
        print("-> PHILOLOGISCHES ERGEBNIS: Das Ignotus-Cluster ist signifikant kohärenter")
        print("   (geringere Streuung). Kinzigs These der Autoreinheit wird gestützt.\n")
    else:
        print("-> PHILOLOGISCHES ERGEBNIS: Das Ignotus-Cluster ist stark heterogen.")
        print("   Kinzigs These der Autoreinheit ist auf Basis dieser Metrik fraglich.\n")

    print("--- TEIL 2: INTER-CLUSTER-DISTANZ (Identifikation mit Amaseia) ---")
    print(f"Mittelwert Ignotus zu Amaseia:      {mean_ignotus:.4f}")
    print(f"Mittelwert Restkorpus zu Amaseia:   {mean_other:.4f}")
    print(f"Definierter Asterius-Schwellenwert: {ASTERIUS_THRESHOLD:.4f}\n")

    if mean_ignotus <= ASTERIUS_THRESHOLD:
        print(f"-> BEWEISFÜHRUNG ERFOLGREICH: Der Ignotus-Mittelwert ({mean_ignotus:.4f}) liegt")
        print("   unterhalb der Demarkationslinie. 'Asterius Ignotus' kann quantitativ")
        print("   als Asterius von Amaseia identifiziert werden.")
    else:
        print(f"-> BEWEISFÜHRUNG GESCHEITERT: Der Ignotus-Mittelwert ({mean_ignotus:.4f}) liegt")
        print("   oberhalb der Demarkationslinie. Die Texte bilden zwar ein in sich")
        print("   geschlossenes Cluster, rücken aber nicht nah genug an den Amasener heran.")
    print("=" * 70)

    # 4. Speichern der aggregierten Werte für die Visualisierung (plot_final_results.py)
    # Zusammenführen und nach Distanz sortieren
    final_df = pd.concat([df_ignotus, df_other]).sort_values(by='Mean_Distance_to_Asterius')
    final_df.to_csv("asterius_final_document_scores.csv", index=False)
    print("\n[INFO] Aggregierte Dokumenten-Scores gespeichert unter 'asterius_final_document_scores.csv'.")


if __name__ == "__main__":
    aggregate_and_evaluate()

def aggregate_inference(input_csv, output_csv):
    print("Lade Inferenz-Ergebnisse...")
    df = pd.read_csv(input_csv)
    
    # 1. Regex: Trenne den Basisnamen des Dokuments vom Sample-Suffix
    # Beispiel: "CPG4610_InPaschaSermo5.xml_0" -> "CPG4610_InPaschaSermo5.xml"
    df['Base_Document'] = df['Pseudo_Text_Titre'].apply(lambda x: re.sub(r'_[0-9]+$', '', str(x)))
    
    # 2. Score für die Aggregation vergeben
    def score_classification(cls):
        if "Core-Asterius" in str(cls):
            return 1.0 # Voller Treffer
        elif "Grauzone" in str(cls):
            return 0.5 # Partieller Treffer (Genre-Rauschen/Verwandtschaft)
        else:
            return 0.0 # Abgewiesen
            
    df['Match_Score'] = df['Classification'].apply(score_classification)
    
    # 3. Gruppieren auf Basisdokument-Ebene
    aggregation = {
        'Distance_to_Centroid': ['mean', 'min', 'max'],
        'Match_Score': ['count', 'mean']
    }
    
    df_agg = df.groupby('Base_Document').agg(aggregation).reset_index()
    
    # Spaltennamen vereinfachen
    df_agg.columns = [
        'Document', 
        'Mean_Distance', 
        'Min_Distance', 
        'Max_Distance', 
        'Total_Samples', 
        'Asterius_Match_Percentage'
    ]
    
    # 4. Prozentrechnung für leichtere Interpretation
    df_agg['Asterius_Match_Percentage'] = (df_agg['Asterius_Match_Percentage'] * 100).round(2)
    df_agg['Mean_Distance'] = df_agg['Mean_Distance'].round(4)
    
    # 5. Sortieren: Höchste Übereinstimmung (Percentage) und kleinste Distanz oben
    df_agg = df_agg.sort_values(by=['Asterius_Match_Percentage', 'Mean_Distance'], ascending=[False, True])
    
    # Exportieren
    df_agg.to_csv(output_csv, index=False)
    print(f"Aggregation abgeschlossen. Gespeichert als {output_csv}\n")
    
    print("--- TOP KANDIDATEN (Gesamtdokument) ---")
    print(df_agg[df_agg['Asterius_Match_Percentage'] >= 50.0][['Document', 'Total_Samples', 'Asterius_Match_Percentage', 'Mean_Distance']].to_string(index=False))

if __name__ == "__main__":
    INFERENCE_FILE = "asterius_inference_results.csv"
    AGGREGATED_FILE = "asterius_final_document_scores.csv"
    aggregate_inference(INFERENCE_FILE, AGGREGATED_FILE)