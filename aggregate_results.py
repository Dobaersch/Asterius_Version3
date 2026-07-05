import pandas as pd
import re

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