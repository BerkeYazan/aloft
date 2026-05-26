import pandas as pd
import os
from tqdm import tqdm
from thefuzz import fuzz

def repair_missing_filename_column(merged_df, extracted_df):
    """
    Repairs a merged_df that is missing the 'Filename' column by matching
    rows back to the extracted_df based on the page text content.
    """
    if 'Filename' in merged_df.columns and not merged_df['Filename'].isnull().all():
        print("Filename column already exists and is populated. No repair needed.")
        return merged_df

    print("Attempting to repair missing 'Filename' column...")
    
    # Create a lookup dictionary: {page_text: filename}
    # This assumes page_text is a reliable unique identifier.
    text_to_filename_map = pd.Series(
        extracted_df['Filename'].values, 
        index=extracted_df['Google Books Page Text']
    ).to_dict()

    # Map the filenames back to the merged dataframe
    merged_df['Filename'] = merged_df['Google Books Page Text'].map(text_to_filename_map)
    
    unmapped_rows = merged_df['Filename'].isnull().sum()
    if unmapped_rows > 0:
        print(f"Warning: Could not find a filename for {unmapped_rows} rows during repair.")
    else:
        print("Successfully repaired and added 'Filename' column to all rows.")
        
    return merged_df

def merge_text():
    # Define file paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extracted_text_path = os.path.join(script_dir, 'extracted_text_google_vision.csv')
    manual_corrections_path = os.path.join(script_dir, 'manual_corrections.csv')
    google_books_path = os.path.join(script_dir, 'google_books_sample_output.csv')
    output_path = os.path.join(script_dir, 'merged_quotes.csv')
    unmerged_log_path = os.path.join(script_dir, 'unmerged_log.csv')
    unmatchable_log_path = os.path.join(script_dir, 'unmatchable_log.csv') # New log file
    fuzzy_log_path = os.path.join(script_dir, 'fuzzy_match_log.csv')
    progress_path = os.path.join(script_dir, 'merging_progress.txt')

    # Load the datasets
    extracted_df = pd.read_csv(extracted_text_path)
    google_books_df = pd.read_csv(google_books_path)

    # Load manual corrections if the file exists
    manual_corrections = {}
    if os.path.exists(manual_corrections_path):
        try:
            corrections_df = pd.read_csv(manual_corrections_path)
            # Create a dictionary for quick lookups: Filename -> Corrected Title
            manual_corrections = pd.Series(corrections_df['Corrected Title'].values, index=corrections_df['Filename']).to_dict()
            print(f"Loaded {len(manual_corrections)} manual corrections.")
        except Exception as e:
            print(f"Warning: Could not load manual corrections file. Error: {e}")

    # --- Continuable Logic ---
    start_row = 0
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                content = f.read().strip()
                if content.isdigit():
                    start_row = int(content)
        except (ValueError, FileNotFoundError):
            start_row = 0

    # Load or create the output dataframe
    if os.path.exists(output_path):
        merged_df = pd.read_csv(output_path)
        # For backward compatibility, drop the old progress column if it exists
        if 'extracted_text_title' in merged_df.columns:
            merged_df = merged_df.drop(columns=['extracted_text_title'])
        # --- Repair Logic ---
        # If the filename column is missing, try to add it back in.
        merged_df = repair_missing_filename_column(merged_df, extracted_df)
    else:
        columns = list(google_books_df.columns) + ['Google Books Page Text', 'Filename']
        merged_df = pd.DataFrame(columns=columns)

    # Load log dataframes
    if os.path.exists(unmerged_log_path):
        unmerged_log_df = pd.read_csv(unmerged_log_path)
    else:
        unmerged_log_df = pd.DataFrame(columns=extracted_df.columns)

    if os.path.exists(unmatchable_log_path):
        unmatchable_log_df = pd.read_csv(unmatchable_log_path)
    else:
        unmatchable_log_df = pd.DataFrame(columns=extracted_df.columns)

    if os.path.exists(fuzzy_log_path):
        fuzzy_log_df = pd.read_csv(fuzzy_log_path)
    else:
        fuzzy_log_df = pd.DataFrame(columns=['OCR Title', 'Likely Title', 'Similarity Score'])

    # --- Merging Logic ---
    new_rows = []
    unmerged_rows = []
    unmatchable_rows = [] # New list for unmatchable entries
    fuzzy_matches_log = []

    # Get the slice of the dataframe to process
    df_to_process = extracted_df.iloc[start_row:]

    if df_to_process.empty:
        print("No new rows to process.")
        # Update progress file even if no new rows, in case file was truncated
        with open(progress_path, 'w') as f:
            f.write(str(len(extracted_df)))
        return

    print(f"Starting merge from row {start_row}")
    for _, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0], desc="Merging text"):
        filename = row['Filename']
        
        # --- Use manual correction if available ---
        if filename in manual_corrections:
            title_to_find = manual_corrections[filename]
        else:
            title_to_find = row['Title']
            
        author_to_find = row['Author']

        # Standardize to handle potential pd.isna() or empty strings
        has_title = isinstance(title_to_find, str) and title_to_find.strip()
        has_author = isinstance(author_to_find, str) and author_to_find.strip()

        # If there's no info to search with, log it and skip.
        if not has_title and not has_author:
            unmatchable_rows.append(row)
            continue
        
        match_found = False
        book_info_to_add = None

        # --- Attempt to match by TITLE first ---
        if has_title:
            # Step 1: Strict Match on Title
            strict_matches = google_books_df[google_books_df['TITLE'].str.startswith(title_to_find, na=False)]
            if not strict_matches.empty:
                book_info_to_add = strict_matches.iloc[0].copy()
                match_found = True
            else:
                # Step 2: Fuzzy Match on Title
                best_score = 0
                best_row = None
                for _, book_row in google_books_df.iterrows():
                    score = fuzz.token_set_ratio(title_to_find, book_row['TITLE'])
                    if score > best_score:
                        best_score, best_row = score, book_row
                
                if best_score > 80:
                    book_info_to_add = best_row.copy()
                    match_found = True
                    fuzzy_matches_log.append({'OCR Title': title_to_find, 'Likely Title': best_row['TITLE'], 'Similarity Score': best_score, 'Match Basis': 'Title'})

        # --- If no title match, attempt to match by AUTHOR ---
        if not match_found and has_author:
            # Step 1: Strict Match on Author
            strict_matches = google_books_df[google_books_df['AUTHOR'] == author_to_find]
            if not strict_matches.empty:
                book_info_to_add = strict_matches.iloc[0].copy()
                match_found = True
            else:
                # Step 2: Fuzzy Match on Author
                best_score = 0
                best_row = None
                for _, book_row in google_books_df.iterrows():
                    score = fuzz.token_set_ratio(author_to_find, book_row['AUTHOR'])
                    if score > best_score:
                        best_score, best_row = score, book_row
                
                if best_score > 80:
                    book_info_to_add = best_row.copy()
                    match_found = True
                    fuzzy_matches_log.append({'OCR Title': title_to_find, 'OCR Author': author_to_find, 'Likely Title': best_row['TITLE'], 'Similarity Score': best_score, 'Match Basis': 'Author'})

        # --- Finalize row based on whether a match was found ---
        if match_found and book_info_to_add is not None:
            book_info_to_add['Google Books Page Text'] = row['Google Books Page Text']
            book_info_to_add['Filename'] = row['Filename'] # Ensure filename is always added
            new_rows.append(book_info_to_add)
        else:
            unmerged_rows.append(row)

    # --- Save results ---
    if new_rows:
        new_data_df = pd.DataFrame(new_rows)
        merged_df = pd.concat([merged_df, new_data_df], ignore_index=True)
        merged_df.to_csv(output_path, index=False)
        print(f"Added {len(new_rows)} new rows to {output_path}")
    else:
        print("No new rows to add.")
        
    if fuzzy_matches_log:
        new_fuzzy_log_df = pd.DataFrame(fuzzy_matches_log)
        fuzzy_log_df = pd.concat([fuzzy_log_df, new_fuzzy_log_df], ignore_index=True)
        fuzzy_log_df.to_csv(fuzzy_log_path, index=False)
        print(f"Logged {len(fuzzy_matches_log)} new fuzzy matches to {fuzzy_log_path}")

    if unmerged_rows:
        new_unmerged_df = pd.DataFrame(unmerged_rows)
        unmerged_log_df = pd.concat([unmerged_log_df, new_unmerged_df], ignore_index=True)
        unmerged_log_df.to_csv(unmerged_log_path, index=False)
        print(f"Logged {len(unmerged_rows)} unmerged rows to {unmerged_log_path}")

    if unmatchable_rows:
        new_unmatchable_df = pd.DataFrame(unmatchable_rows)
        unmatchable_log_df = pd.concat([unmatchable_log_df, new_unmatchable_df], ignore_index=True)
        unmatchable_log_df.to_csv(unmatchable_log_path, index=False)
        print(f"Logged {len(unmatchable_rows)} unmatchable rows (no title/author) to {unmatchable_log_path}")
    
    # Update progress file
    with open(progress_path, 'w') as f:
        f.write(str(len(extracted_df)))
    print(f"Progress updated. Processed up to row {len(extracted_df)}.")

if __name__ == '__main__':
    merge_text() 