import pandas as pd
import os

def calculate_character_retention(original_file, cleaned_file, column_name):
    """
    Calculates the percentage of characters retained in a cleaned file compared to the original.

    Args:
        original_file (str): Path to the original CSV file.
        cleaned_file (str): Path to the cleaned CSV file.
        column_name (str): The name of the column to compare.

    Returns:
        float: The percentage of characters retained, or None if an error occurs.
    """
    if not os.path.exists(original_file):
        print(f"Error: Original file not found at '{original_file}'")
        return None
    if not os.path.exists(cleaned_file):
        print(f"Error: Cleaned file not found at '{cleaned_file}'")
        return None

    print("Loading original and cleaned datasets...")
    try:
        original_df = pd.read_csv(original_file)
        cleaned_df = pd.read_csv(cleaned_file)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None

    if column_name not in original_df.columns:
        print(f"Error: Column '{column_name}' not found in the original file.")
        return None
    if column_name not in cleaned_df.columns:
        print(f"Error: Column '{column_name}' not found in the cleaned file.")
        return None

    # Calculate total characters in the original text column
    original_chars = original_df[column_name].dropna().apply(lambda x: len(str(x))).sum()
    
    # Calculate total characters in the cleaned text column
    cleaned_chars = cleaned_df[column_name].dropna().apply(lambda x: len(str(x))).sum()

    if original_chars == 0:
        print("Warning: The original column has no text.")
        return 0.0 if cleaned_chars == 0 else float('inf')

    retention_percentage = (cleaned_chars / original_chars) * 100

    print(f"\n--- Comparison for column: '{column_name}' ---")
    print(f"Total characters in original file: {original_chars}")
    print(f"Total characters in cleaned file: {cleaned_chars}")
    print(f"Percentage of characters retained: {retention_percentage:.2f}%")
    
    return retention_percentage

if __name__ == '__main__':
    original_csv = 'data/interim/google_books_work/merged_quotes.csv'
    cleaned_csv = 'data/interim/google_books_work/quotes_cleaned.csv'
    column_to_compare = 'Google Books Page Text'
    
    calculate_character_retention(original_csv, cleaned_csv, column_to_compare) 