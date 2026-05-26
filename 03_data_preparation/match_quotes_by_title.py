import pandas as pd
from thefuzz import fuzz
import os

def find_cross_file_fuzzy_matches(file1_path, file2_path, output_path, file1_title_col, file2_title_col, file1_text_col, file2_quote_col, threshold=90):
    """
    Finds fuzzy matches for quotes between two CSV files, conditioned on matching titles.

    Args:
        file1_path (str): Path to the first CSV file (e.g., with page text).
        file2_path (str): Path to the second CSV file (e.g., with quotes).
        output_path (str): Path to save the output CSV file with matches.
        file1_title_col (str): The column name for the title in the first file.
        file2_title_col (str): The column name for the title in the second file.
        file1_text_col (str): The column name for the text to search within in the first file.
        file2_quote_col (str): The column name for the quote to search for in the second file.
        threshold (int, optional): The fuzzy matching threshold (0-100). Defaults to 90.
    """
    print(f"Reading data from {file1_path} and {file2_path}...")
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    # Create temporary, normalized title columns for matching
    df1_normalized_title_col = 'normalized_' + file1_title_col
    df2_normalized_title_col = 'normalized_' + file2_title_col
    
    df1[df1_normalized_title_col] = df1[file1_title_col].str.lower().str.strip()
    df2[df2_normalized_title_col] = df2[file2_title_col].str.lower().str.strip()

    # Drop rows with missing titles or text/quotes
    df1.dropna(subset=[file1_title_col, file1_text_col], inplace=True)
    df2.dropna(subset=[file2_title_col, file2_quote_col], inplace=True)

    # Add a unique identifier to df1 to group by later
    df1['original_index'] = df1.index

    # Group the second dataframe by the normalized title for efficient lookup
    quotes_by_title = df2.groupby(df2_normalized_title_col)[file2_quote_col].apply(list)

    matches = []
    print("Finding matches across files...")
    # Iterate over the first dataframe
    for index, row1 in df1.iterrows():
        title = row1[df1_normalized_title_col]
        page_text = str(row1[file1_text_col])

        # Check if the title exists in the grouped quotes
        if title in quotes_by_title:
            # Iterate through all quotes for the matching title
            for quote in quotes_by_title[title]:
                score = fuzz.token_set_ratio(page_text, str(quote))
                if score >= threshold:
                    # If a match is found, append the relevant data to our list
                    match_data = row1.to_dict()
                    match_data['matched_quote'] = quote
                    match_data['match_score'] = score
                    matches.append(match_data)
                    
    if matches:
        matched_df = pd.DataFrame(matches)
        
        # --- NEW: Group matches by the original text ---
        print(f"Found {len(matched_df)} total matches. Now grouping by page...")
        
        # Define a custom aggregation
        def aggregate_matches(group):
            # Use a unique, unlikely separator for the list of quotes
            separator = '|||'
            # Join all matched quotes into a single string
            all_quotes = separator.join(group['matched_quote'])
            # Take the first row's data for all other columns
            d = group.iloc[0].to_dict()
            d['matched_quotes_list'] = all_quotes
            return pd.Series(d)

        # Group by the unique identifier of the original row
        grouped_df = matched_df.groupby('original_index').apply(aggregate_matches).reset_index(drop=True)

        # Clean up columns we no longer need for the output
        grouped_df.drop(columns=['matched_quote', 'match_score', df1_normalized_title_col, 'original_index'], inplace=True, errors='ignore')

        print(f"Grouped into {len(grouped_df)} unique pages. Saving to {output_path}...")
        grouped_df.to_csv(output_path, index=False)
        print("Done.")
    else:
        print("No matches found.")

if __name__ == "__main__":
    # Configuration
    FILE1_PATH = "data/interim/google_books_work/quotes_corrected.csv"
    FILE2_PATH = "data/interim/cleaning_data/Data/goodreads-english-popular-quotes.csv"
    OUTPUT_FILE = "data/interim/google_books_work/cross_file_fuzzy_matches.csv"

    # Column names from inspection
    FILE1_TITLE_COL = "TITLE"
    FILE2_TITLE_COL = "TITLE"
    FILE1_TEXT_COL = "Google Books Page Text"
    FILE2_QUOTE_COL = "QUOTE"
    
    CONFIDENCE_THRESHOLD = 90

    find_cross_file_fuzzy_matches(
        FILE1_PATH, FILE2_PATH, OUTPUT_FILE,
        FILE1_TITLE_COL, FILE2_TITLE_COL,
        FILE1_TEXT_COL, FILE2_QUOTE_COL,
        threshold=CONFIDENCE_THRESHOLD
    ) 