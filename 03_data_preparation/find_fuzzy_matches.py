import pandas as pd
from thefuzz import fuzz

def find_fuzzy_matches(input_path, output_path, column1, column2, threshold=90):
    """
    Finds fuzzy matches between two columns in a CSV file and saves the matches to a new CSV.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to save the output CSV file.
        column1 (str): The name of the first column to compare.
        column2 (str): The name of the second column to compare.
        threshold (int, optional): The fuzzy matching threshold (0-100). Defaults to 90.
    """
    print(f"Reading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return

    # Ensure the specified columns exist in the DataFrame
    if column1 not in df.columns or column2 not in df.columns:
        print(f"Error: One or both of the specified columns ('{column1}', '{column2}') not found in the CSV.")
        return
        
    # Drop rows where either of the key columns is NaN
    df.dropna(subset=[column1, column2], inplace=True)

    def is_match(row):
        # Using token_set_ratio for partial matches with different word order
        score = fuzz.token_set_ratio(str(row[column1]), str(row[column2]))
        return score >= threshold

    print("Finding matches...")
    # Apply the matching function row-wise
    matched_rows = df[df.apply(is_match, axis=1)]

    if not matched_rows.empty:
        print(f"Found {len(matched_rows)} matches. Saving to {output_path}...")
        matched_rows.to_csv(output_path, index=False)
        print("Done.")
    else:
        print("No matches found.")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data/interim/google_books_work/quotes_cleaned.csv"
    OUTPUT_FILE = "data/interim/google_books_work/fuzzy_matches.csv"
    COLUMN_1 = "QUOTE"
    COLUMN_2 = "Google Books Page Text"
    CONFIDENCE_THRESHOLD = 90  # Adjust this value based on desired strictness (0-100)

    find_fuzzy_matches(INPUT_FILE, OUTPUT_FILE, COLUMN_1, COLUMN_2, threshold=CONFIDENCE_THRESHOLD) 