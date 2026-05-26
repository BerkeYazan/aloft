import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """Loads the CSV data from the given path."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return None
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def get_metric_columns(df):
    """Categorizes and returns only the metric columns."""
    all_columns = df.columns.tolist()
    
    text_keywords = ['Quote', 'Snippet', 'Baseline', 'Context', 'Text']
    text_columns = [col for col in all_columns if any(keyword in col for keyword in text_keywords) and df[col].dtype == 'object']
    
    metadata_keywords = [
        'Likes of', 'Author of', 'Title of', 'Language of', 'Publication Date',
        'Genre of', 'language_normalized', 'quote_length', 'ocr_page_length',
        'matched_snippet_length'
    ]
    metadata_columns = [col for col in all_columns if any(keyword in col for keyword in metadata_keywords)]
    
    metric_columns = [col for col in all_columns if col not in text_columns and col not in metadata_columns]
    
    return metric_columns

def analyze_correlation_with_likes(df, metric_columns, target_col='Likes of Sample Quote'):
    """Calculates and prints the correlation of metrics with a target column."""
    print(f"\n--- Correlation Analysis with '{target_col}' ---")

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return

    # Determine the correct metric prefix based on the target column
    if 'Sample' in target_col:
        prefix = 'sample_'
    elif 'Popular' in target_col:
        prefix = 'popular_'
    else:
        print(f"Cannot determine metric prefix for target column: {target_col}")
        return

    # Filter for metrics corresponding to the target column
    relevant_metrics = [col for col in metric_columns if col.startswith(prefix)]
    
    # Select only numeric metric columns for correlation
    numeric_metric_columns = df[relevant_metrics].select_dtypes(include=['number']).columns.tolist()

    if not numeric_metric_columns:
        print(f"No numeric metrics found with prefix '{prefix}'")
        return

    correlations = df[numeric_metric_columns].corrwith(df[target_col], method='spearman')
    correlations = correlations.dropna().sort_values(ascending=False)

    print(f"\nCorrelations for metrics starting with '{prefix}':")
    print(correlations)
    print("\n" + "="*50 + "\n")


def generate_correlation_heatmap(df, output_path):
    """Generates and saves a correlation heatmap for sample quote metrics."""
    print("--- Generating Correlation Heatmap for Sample Quote Metrics ---")
    
    sample_metrics = [col for col in df.columns if col.startswith('sample_')]
    
    # Select only numeric columns among sample_metrics
    numeric_sample_metrics = df[sample_metrics].select_dtypes(include=['number']).columns.tolist()

    if not numeric_sample_metrics:
        print("No numeric metrics starting with 'sample_' found.")
        return

    # Create a cleaner name mapping for the plot
    clean_names = {col: col.replace('sample_', '').replace('_', ' ').title() for col in numeric_sample_metrics}
    
    subset_df = df[numeric_sample_metrics].rename(columns=clean_names)
    
    # Calculate correlation matrix
    corr_matrix = subset_df.corr(method='spearman')

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
    plt.title('Spearman Correlation Matrix of Sample Quote Metrics', size=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_path)
    print(f"Heatmap saved to '{output_path}'")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    csv_file_path = "data/outputs/master_metrics/ALOFT_master_metrics.csv"
    heatmap_output_path = "data/outputs/master_metrics/correlation_analysis/sample_quote_metric_correlations.png"

    df = load_data(csv_file_path)

    if df is not None:
        metric_cols = get_metric_columns(df)
        analyze_correlation_with_likes(df, metric_cols, target_col='Likes of Sample Quote')
        analyze_correlation_with_likes(df, metric_cols, target_col='Likes of Popular Quote')
        generate_correlation_heatmap(df, heatmap_output_path) 