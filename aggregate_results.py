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

    # Extract base document title by removing the chunk index suffix (e.g., "_0", "_1")
    def extract_doc_name(title):
        return re.sub(r'_[0-9]+$', '', str(title))

    df['Base_Document'] = df['Pseudo_Text_Titre'].apply(extract_doc_name)

    # The threshold is constant across all chunks for the Asterius centroid
    asterius_threshold = df['Threshold'].iloc[0]

    # Aggregate chunk data to document level
    doc_groups = df.groupby('Base_Document')
    results = []

    for doc_name, group in doc_groups:
        total_chunks = len(group)
        mean_distance = group['Distance_to_Centroid'].mean()

        # Calculate the percentage of chunks strictly within the stylistic threshold
        asterius_chunks = group[group['Distance_to_Centroid'] <= asterius_threshold]
        match_percentage = (len(asterius_chunks) / total_chunks) * 100.0

        results.append({
            'Document': doc_name,
            'Total_Chunks': total_chunks,
            'Mean_Distance_to_Asterius': mean_distance,
            'Asterius_Match_Percentage': match_percentage,
            'Threshold': asterius_threshold
        })

    df_agg = pd.DataFrame(results)

    # Sort by highest match percentage, then lowest distance (stylistic proximity)
    df_agg = df_agg.sort_values(by=['Asterius_Match_Percentage', 'Mean_Distance_to_Asterius'], ascending=[False, True])
    df_agg.to_csv(output_csv, index=False)

    print(f"[Success] Aggregated document scores saved to '{output_csv}'")

    print("\n--- TOP CANDIDATES (Overall Document Match >= 50%) ---")
    top_candidates = df_agg[df_agg['Asterius_Match_Percentage'] >= 50.0]

    if not top_candidates.empty:
        print(top_candidates[['Document', 'Asterius_Match_Percentage', 'Mean_Distance_to_Asterius']].to_string(
            index=False))
    else:
        print("No documents met the 50% threshold for Asterius authorship.")

    # --- Statistical Isolation Check (Cohen's d) ---
    # Evaluates if the newly found Asterius cluster is statistically distinct from the rejected corpus
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
            print(
                "[✗] Weak Isolation: The candidates overlap significantly with the rest of the corpus (Methodological Warning).")
    else:
        print(
            "\n[Info] Not enough data points (documents) in either the accepted or rejected cluster to calculate Cohen's d.")


if __name__ == "__main__":
    aggregate_and_evaluate()