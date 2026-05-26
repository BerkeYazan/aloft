import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

def load_and_restructure_data_v2(file_path):
    """
    Loads data and restructures it for classification with a more robust
    and explicit mapping to avoid data leakage.
    """
    print("--- Loading and Restructuring Data (v2) ---")
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Define a more explicit mapping for each source
    source_configs = {
        'Goodreads Sample Quote': {
            'prefix': 'sample_',
            'is_extraordinary': 1
        },
        'Goodreads Popular Quote': {
            'prefix': 'popular_',
            'is_extraordinary': 1
        },
        'T50 Quote': {
            'prefix': 't50_quote_',
            'is_extraordinary': 1,
            'special_cases': {'t50_flesch': 'flesch', 't50_cl': 'cl', 't50_pmi': 'pmi', 't50_entropy': 'entropy', 't50_surprisal': 'surprisal'}
        },
        'Google Books Length Matched Snippet': {
            'prefix': 'snippet_',
            'is_extraordinary': 0
        },
        'T50 Quote-Free Context Length Matched': {
            'prefix': 't50freelength_',
            'is_extraordinary': 0,
             'special_cases': {'t50_quote_free_context_length_matched_lex_div': 'lex_div'}
        },
        'Non-Literary Baseline': {
            'prefix': 'nonlit_',
            'is_extraordinary': 0
        }
    }

    generic_feature_names = [
        'flesch', 'cl', 'lex_div', 'entropy', 'pmi', 'surprisal',
        'sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'sentiment_polarity'
    ]

    all_processed_data = []

    for source_name, config in source_configs.items():
        source_df = pd.DataFrame()
        prefix = config['prefix']
        
        # Handle regular prefixed columns
        for generic_name in generic_feature_names:
            source_col = f"{prefix}{generic_name}"
            if source_col in df.columns:
                source_df[generic_name] = df[source_col]
        
        # Handle special cases where naming is irregular
        if 'special_cases' in config:
            for source_col, generic_name in config['special_cases'].items():
                 if source_col in df.columns:
                    source_df[generic_name] = df[source_col]

        # Skip if no data was found for this source
        if source_df.empty:
            print(f"Warning: No columns found for source: '{source_name}'. Skipping.")
            continue

        source_df['is_extraordinary'] = config['is_extraordinary']
        all_processed_data.append(source_df)
        print(f"Processed source: '{source_name}' (Class: {'Extraordinary' if config['is_extraordinary'] else 'Ordinary'})")

    final_df = pd.concat(all_processed_data, ignore_index=True)
    
    # Impute missing values with the median for the entire combined dataset
    for col in final_df.columns:
        if final_df[col].isnull().any():
            median_val = final_df[col].median()
            final_df[col] = final_df[col].fillna(median_val)
            
    print(f"\nFinal dataset created with {final_df.shape[0]} samples and {final_df.shape[1]-1} features.")
    print(f"Class distribution:\n{final_df['is_extraordinary'].value_counts()}")
    
    return final_df


def perform_shap_classification_analysis(df, output_dir):
    """
    Trains a classifier, performs SHAP analysis, and saves plots.
    """
    print("\n--- Starting SHAP Classification Analysis ---")
    
    X = df.drop('is_extraordinary', axis=1)
    y = df['is_extraordinary']

    # Train a LightGBM classifier
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)
    print("Classifier training complete.")

    # Create a SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    print("SHAP values calculated.")

    # Create cleaner names for plotting
    clean_names = {col: col.replace('_', ' ').title() for col in X.columns}
    X.columns = X.columns.map(clean_names)
    
    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    # Generate and save SHAP summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=X.shape[1])
    plt.title(f'SHAP Summary for Distinguishing Extraordinary Text', size=14)
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'classification_shap_summary_plot_v2.png')
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to '{summary_plot_path}'")
    
    # Generate and save SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=X.shape[1])
    plt.title(f'Feature Importance for Distinguishing Extraordinary Text', size=14)
    plt.tight_layout()
    bar_plot_path = os.path.join(output_dir, 'classification_shap_feature_importance_bar_plot_v2.png')
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"SHAP feature importance bar plot saved to '{bar_plot_path}'")

    print("\n--- SHAP Analysis Complete ---")


if __name__ == "__main__":
    csv_file_path = "data/outputs/master_metrics/ALOFT_master_metrics.csv"
    shap_output_path = "data/outputs/master_metrics/shap_classification_analysis/"
    
    unified_df = load_and_restructure_data_v2(csv_file_path)

    if unified_df is not None and not unified_df.empty:
        perform_shap_classification_analysis(unified_df, shap_output_path) 