import pandas as pd
import numpy as np
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def aggregate_and_evaluate(input_csv="asterius_inference_results.csv", output_csv="asterius_final_document_scores.csv"):
    """
    Aggregates chunk-level distances to document-level scores and computes
    the statistical isolation of the identified target cluster.
    """
    print("--- Starting Document Aggregation and Statistical Evaluation ---")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[Error] The file '{input_csv}' was not found. Run the inference script first.")
        return

    # 1. Extract base document title by removing the chunk index suffix (e.g., "_0", "_1")
    def extract_doc_name(title):
        return re.sub(r'_[0-9]+$', '', str(title))

    # MODIFIED: Changed 'Pseudo_Text_Titre' to the actual exported column 'Document'
    df['Base_Document'] = df['Document'].apply(extract_doc_name)

    # MODIFIED: Set the threshold directly to match our 8D Euclidean space limit
    asterius_threshold = 2.5000

    # 3. Aggregate chunk data to document level
    doc_groups = df.groupby('Base_Document')
    results = []

    for doc_name, group in doc_groups:
        total_chunks = len(group)
        mean_distance = group['Distance_to_Centroid'].mean()

        # Calculate the percentage of chunks strictly within the stylistic threshold
        asterius_chunks = group[group['Distance_to_Centroid'] <= asterius_threshold]
        match_percentage = (len(asterius_chunks) / total_chunks) * 100

        # Classify the entire document based on its average distance
        if mean_distance <= asterius_threshold:
            final_class = "Core-Asterius"
        elif mean_distance <= (asterius_threshold + 1.0):
            final_class = "Grey Zone (Mixed/Edited)"
        else:
            final_class = "Foreign Author"

        results.append({
            'Document': doc_name,
            'Total_Chunks': total_chunks,
            'Asterius_Chunks': len(asterius_chunks),
            'Asterius_Match_Percentage': round(match_percentage, 2),
            'Mean_Distance_to_Asterius': round(mean_distance, 4),
            'Final_Classification': final_class
        })

    # Sort results to have the best Asterius candidates at the top
    df_agg = pd.DataFrame(results).sort_values(by='Mean_Distance_to_Asterius')
    df_agg.to_csv(output_csv, index=False)
    print(f"[Success] Aggregated results saved to '{output_csv}'")

    # 4. Statistical Isolation Check (Cohen's d)
    top_candidates = df_agg[df_agg['Asterius_Match_Percentage'] >= 50.0]
    rejected_docs = df_agg[df_agg['Asterius_Match_Percentage'] < 50.0]

    if not top_candidates.empty and len(top_candidates) > 1 and len(rejected_docs) > 1:
        mean_top = top_candidates['Mean_Distance_to_Asterius'].mean()
        mean_rej = rejected_docs['Mean_Distance_to_Asterius'].mean()

        var_top = top_candidates['Mean_Distance_to_Asterius'].var(ddof=1)
        var_rej = rejected_docs['Mean_Distance_to_Asterius'].var(ddof=1)

        n_top = len(top_candidates)
        n_rej = len(rejected_docs)

        # Pooled standard deviation calculation
        pooled_std = np.sqrt(((n_top - 1) * var_top + (n_rej - 1) * var_rej) / (n_top + n_rej - 2))
        cohens_d = abs(mean_top - mean_rej) / pooled_std

        print("\n--- STATISTICAL ISOLATION (Cohen's d) ---")
        print(f"Effect Size (Cohen's d): {cohens_d:.4f}")

        if cohens_d > 0.8:
            print("[✓] Valid Isolation: The Asterius candidates are strongly separated from the rest of the corpus.")
        else:
            print("[✗] Weak Isolation: The candidates overlap significantly with the rest of the corpus.")


if __name__ == "__main__":
    aggregate_and_evaluate()