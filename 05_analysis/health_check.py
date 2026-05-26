import pandas as pd
import os

def perform_health_check(file_path):
    """
    Performs a health check on a given CSV file path.
    Outputs the head, info, NA values, and descriptive statistics.
    """
    print(f"--- Starting health check for: {file_path} ---\n")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        directory = os.path.dirname(file_path)
        if os.path.isdir(directory):
            print(f"Files in '{directory}':")
            try:
                for f in os.listdir(directory):
                    print(f"- {f}")
            except OSError as e:
                print(f"Could not list files in directory: {e}")
        return

    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- Basic Information ---
    print("--- DataFrame Head ---")
    print(df.head())
    print("\n" + "="*50 + "\n")

    print("--- DataFrame Info ---")
    # Using a buffer to capture the output of df.info() to print it
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)
    print("\n" + "="*50 + "\n")

    # --- NA Values ---
    print("--- NA Values per Column ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        na_counts = df.isnull().sum()
        print(na_counts)
    print("\n" + "="*50 + "\n")
    
    # --- Duplicates ---
    print("--- Duplicate Rows ---")
    duplicate_rows = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")
    print("\n" + "="*50 + "\n")


    # --- Descriptive Statistics ---
    print("--- Descriptive Statistics ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.describe(include='all'))
    print("\n" + "="*50 + "\n")

    categorize_columns(df)

    print("--- Health check complete ---")


def map_metrics_to_text(text_columns, metric_columns):
    """
    Maps metric columns to their corresponding text source columns based on naming conventions.
    """
    # Define the mapping from a prefix to a text column name.
    # Longer, more specific prefixes are listed first to ensure they are matched correctly.
    prefix_map = {
        't50_quote_free_context_length_matched': 'T50 Quote-Free Context Length Matched',
        't50freelength': 'T50 Quote-Free Context Length Matched',
        't50_quote_free_context': 'T50 Quote-Free Context',
        't50free': 'T50 Quote-Free Context',
        't50_quote': 'T50 Quote',
        't50': 'T50 Quote',
        'matched_snippet': 'Google Books Length Matched Snippet',
        'snippet': 'Google Books Length Matched Snippet',
        'page_text': 'Google Books Page Text',
        'page': 'Google Books Page Text',
        'nonlit_baseline': 'Non-Literary Baseline',
        'nonlit': 'Non-Literary Baseline',
        'sample': 'Goodreads Sample Quote',
        'popular': 'Goodreads Popular Quote',
    }

    # Initialize the final mapping structure.
    # Keys are text columns, values are lists of associated metric columns.
    final_mapping = {text_col: [] for text_col in text_columns}
    unmapped_metrics = []

    for metric in metric_columns:
        mapped = False
        # Find the longest matching prefix for each metric.
        for prefix, text_column_name in prefix_map.items():
            if metric.startswith(prefix + '_'):
                if text_column_name in final_mapping:
                    final_mapping[text_column_name].append(metric)
                    mapped = True
                    break  # Stop after finding the first (and longest) match
        
        if not mapped:
            unmapped_metrics.append(metric)

    return final_mapping, unmapped_metrics


def categorize_columns(df):
    """
    Categorizes columns into text, metadata, and metric columns.
    """
    
    all_columns = df.columns.tolist()

    text_columns = [c for c in all_columns if df[c].dtype == 'object' and any(keyword in c for keyword in ['Quote', 'Snippet', 'Baseline', 'Context', 'Text'])]
    
    # Identify text columns based on common keywords
    text_keywords = ['Quote', 'Snippet', 'Baseline', 'Context', 'Text']
    text_columns = [col for col in all_columns if any(keyword in col for keyword in text_keywords) and df[col].dtype == 'object']


    # Define specific metadata columns
    metadata_keywords = [
        'Likes of', 'Author of', 'Title of', 'Language of', 'Publication Date',
        'Genre of', 'language_normalized', 'quote_length', 'ocr_page_length',
        'matched_snippet_length'
    ]
    metadata_columns = [col for col in all_columns if any(keyword in col for keyword in metadata_keywords)]


    # Metric columns are the remaining columns
    metric_columns = [
        col for col in all_columns 
        if col not in text_columns and col not in metadata_columns
    ]

    print("--- Column Categories ---")
    
    print(f"\n--- Text Columns ({len(text_columns)}) ---")
    for col in text_columns:
        print(f"- {col}")

    print(f"\n--- Metadata Columns ({len(metadata_columns)}) ---")
    for col in metadata_columns:
        print(f"- {col}")

    print(f"\n--- Metric Columns ({len(metric_columns)}) ---")
    for col in metric_columns:
        print(f"- {col}")

    print("\n" + "="*50 + "\n")
    
    # Now map metrics to text sources
    metric_map, unmapped_metrics = map_metrics_to_text(text_columns, metric_columns)

    print("\n--- Metric to Text Column Mapping ---")
    for text_col, metrics in metric_map.items():
        if metrics:  # Only print if there are associated metrics
            print(f"\n'{text_col}' is associated with {len(metrics)} metrics:")
            for metric in sorted(metrics):
                print(f"  - {metric}")

    if unmapped_metrics:
        print(f"\n--- Unmapped Metrics ({len(unmapped_metrics)}) ---")
        for metric in sorted(unmapped_metrics):
            print(f"  - {metric}")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # File path provided by the user
    csv_file_path = "data/outputs/master_metrics/ALOFT_master_metrics.csv"
    perform_health_check(csv_file_path) 