import pandas as pd
import numpy as np
import re


def aggregate_and_evaluate(input_csv, output_csv):
    # 1. Daten laden
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[Fehler] Die Datei '{input_csv}' wurde nicht gefunden.")
        return

    # Nur Vergleiche mit dem Zielautor "Asterius" filtern
    df_asterius = df[df['ComparatorClass'].str.lower() == 'asterius'].copy()

    if df_asterius.empty:
        print("[Fehler] Keine Vergleiche für 'Asterius' in den Daten gefunden.")
        return

    ASTERIUS_THRESHOLD = df_asterius['Dynamic_Threshold'].iloc[0]

    # 2. Document-Level Aggregation (Zusammenfassen der Chunks)
    def extract_doc_name(sample_label):
        # Entfernt das Suffix "_0", "_1" etc.
        return re.sub(r'_[0-9]+$', '', str(sample_label))

    df_asterius['Base_Document'] = df_asterius['Pseudo_Text_Titre'].apply(extract_doc_name)

    # Score für die prozentuale Übereinstimmung vergeben
    def score_classification(cls):
        if "Core-Asterius" in str(cls):
            return 1.0  # Voller Treffer
        elif "Grauzone" in str(cls):
            return 0.5  # Partieller Treffer
        else:
            return 0.0  # Abgewiesen

    df_asterius['Match_Score'] = df_asterius['Classification'].apply(score_classification)

    aggregation = {
        'Distance_to_Centroid': ['mean', 'min', 'max'],
        'Match_Score': ['count', 'mean']
    }

    df_agg = df_asterius.groupby('Base_Document').agg(aggregation).reset_index()

    df_agg.columns = [
        'Document',
        'Mean_Distance_to_Asterius',
        'Min_Distance',
        'Max_Distance',
        'Total_Samples',
        'Asterius_Match_Percentage'
    ]

    # Prozentrechnung
    df_agg['Asterius_Match_Percentage'] = (df_agg['Asterius_Match_Percentage'] * 100).round(2)

    # 3. Cluster Separation: Erweiterte statistische Isolation ("Asterius Ignotus" vs. "Restliches Korpus")
    ignotus_mask = df_agg['Document'].str.contains('AsteriusIgnotus', case=False, na=False)

    df_ignotus = df_agg[ignotus_mask]
    df_other = df_agg[~ignotus_mask]

    n_ignotus = len(df_ignotus)
    n_other = len(df_other)

    # Berechnung der Varianzen und Standardabweichungen (Intra-Cluster-Kohärenz) mit ddof=1 für Stichproben
    var_ignotus = df_ignotus['Mean_Distance_to_Asterius'].var(ddof=1) if n_ignotus > 1 else 0
    std_ignotus = df_ignotus['Mean_Distance_to_Asterius'].std(ddof=1) if n_ignotus > 1 else 0

    var_other = df_other['Mean_Distance_to_Asterius'].var(ddof=1) if n_other > 1 else 0
    std_other = df_other['Mean_Distance_to_Asterius'].std(ddof=1) if n_other > 1 else 0

    # Berechnung der Mittelwerte (Inter-Cluster-Distanz)
    mean_ignotus = df_ignotus['Mean_Distance_to_Asterius'].mean()
    mean_other = df_other['Mean_Distance_to_Asterius'].mean()

    # Berechnung der Effektstärke (Cohen's d) für Cluster-Isolation
    if n_ignotus > 1 and n_other > 1:
        pooled_var = ((n_ignotus - 1) * var_ignotus + (n_other - 1) * var_other) / (n_ignotus + n_other - 2)
        cohens_d = abs(mean_ignotus - mean_other) / np.sqrt(pooled_var)
    else:
        cohens_d = 0.0

    # --- REPORT GENERIERUNG ---
    print("=" * 80)
    print(" MAKRO-ANALYSE: ASTERIUS IGNOTUS CLUSTER VS. PSEUDO-KORPUS")
    print("=" * 80)
    print(f"Anzahl evaluierter 'Asterius Ignotus' Dokumente: {n_ignotus}")
    print(f"Anzahl restlicher Pseudo-Dokumente:              {n_other}\n")

    print("--- TEIL 1: INTRA-CLUSTER-KOHÄRENZ (Homogenität) ---")
    print(f"Streuung (StdAbw) Asterius Ignotus: {std_ignotus:.4f}")
    print(f"Streuung (StdAbw) Restliches Korpus:{std_other:.4f}\n")

    print("--- TEIL 2: INTER-CLUSTER-DISTANZ (Identifikation & Isolation) ---")
    print(f"Mittelwert Ignotus zu Amaseia:      {mean_ignotus:.4f}")
    print(f"Mittelwert Restkorpus zu Amaseia:   {mean_other:.4f}")
    print(f"Importierter ML-Schwellenwert:      {ASTERIUS_THRESHOLD:.4f}\n")

    print("--- TEIL 3: EMPIRISCHE BEWEISFÜHRUNG ---")

    # 1. Check: Ist es Asterius?
    if mean_ignotus <= ASTERIUS_THRESHOLD:
        print(f"[✓] Zuweisung valide: Ignotus liegt innerhalb des Asterius-Schwellenwerts.")
    else:
        print(f"[✗] Zuweisung fraglich: Ignotus liegt außerhalb des Asterius-Schwellenwerts.")

    # 2. Check: Ist das Cluster isoliert? (Cohen's d > 0.8 gilt als starker Effekt)
    print(f"Distanz-Effektstärke (Cohen's d):   {cohens_d:.4f}")
    if cohens_d > 0.8:
        print("[✓] Isolation valide: Das Ignotus-Cluster grenzt sich deutlich und")
        print("    vom restlichen pseudo-chrysostomischen Material ab.")
    else:
        print("[✗] Isolation schwach: Das Ignotus-Cluster überschneidet sich stark mit")
        print("    dem Restkorpus. Keine klare philologische Demarkation möglich.")
    print("=" * 80)

    # 4. Ergebnisse sortieren und speichern
    df_agg = df_agg.sort_values(by=['Asterius_Match_Percentage', 'Mean_Distance_to_Asterius'], ascending=[False, True])
    df_agg.to_csv(output_csv, index=False)

    print(f"\n[INFO] Aggregierte Dokumenten-Scores gespeichert unter '{output_csv}'.")
    print("\n--- TOP KANDIDATEN (Gesamtdokument) ---")
    # Ausgabe verschlanken, nur Kandidaten mit mind. 50% Übereinstimmung
    print(df_agg[df_agg['Asterius_Match_Percentage'] >= 50.0][
        ['Document', 'Total_Samples', 'Asterius_Match_Percentage', 'Mean_Distance_to_Asterius']].to_string(
        index=False))


if __name__ == "__main__":
    INFERENCE_FILE = "asterius_inference_results.csv"
    AGGREGATED_FILE = "asterius_final_document_scores.csv"

    aggregate_and_evaluate(INFERENCE_FILE, AGGREGATED_FILE)