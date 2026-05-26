import pandas as pd
import numpy as np

def merge_manual_corrections():
    """
    Merges manually corrected unquoted text back into the main dataset.
    Creates a final dataset with both quoted and unquoted versions.
    """
    
    print("Loading datasets...")
    
    # Load the main dataset (5k+ rows with quotes)
    main_df = pd.read_csv('data/interim/google_books_work/quotes_corrected.csv')
    print(f"Main dataset: {len(main_df)} rows")
    
    # Load manually corrected unquoted text
    manual_df = pd.read_csv('data/interim/google_books_work/unquoted_text_manual_review.csv')
    print(f"Manual corrections: {len(manual_df)} rows")
    
    # Create a lookup for manual corrections using a combination of identifiers
    # We'll use TITLE + AUTHOR + original Google Books Page Text as the key
    manual_lookup = {}
    
    for idx, row in manual_df.iterrows():
        # Create a unique key for matching
        key = (
            str(row.get('TITLE', '')).strip(),
            str(row.get('AUTHOR', '')).strip(), 
            str(row.get('Google Books Page Text', '')).strip()[:100]  # First 100 chars for matching
        )
        manual_lookup[key] = row.get('Google Books Page Text_unquoted', '')
    
    print(f"Created lookup table with {len(manual_lookup)} manual corrections")
    
    # Replace Google Books Page Text with unquoted version where available
    matches_found = 0
    
    for idx, row in main_df.iterrows():
        # Create the same key for this row
        key = (
            str(row.get('TITLE', '')).strip(),
            str(row.get('AUTHOR', '')).strip(),
            str(row.get('Google Books Page Text', '')).strip()[:100]
        )
        
        # If we have a manual correction for this row, replace the original
        if key in manual_lookup:
            main_df.loc[idx, 'Google Books Page Text'] = manual_lookup[key]
            matches_found += 1
        # If no manual correction exists, keep the original text as-is
    
    print(f"Successfully merged {matches_found} manual corrections")
    print(f"Remaining rows use original text: {len(main_df) - matches_found}")
    
    # Save the final merged dataset
    output_file = 'data/interim/google_books_work/quotes_final_cleaned.csv'
    main_df.to_csv(output_file, index=False)
    print(f"\nFinal dataset saved to: {output_file}")
    print(f"Total rows: {len(main_df)}")
    
    # Create summary statistics
    print("\n" + "="*50)
    print("MERGE SUMMARY")
    print("="*50)
    
    print(f"Rows with manual corrections applied: {matches_found}")
    print(f"Rows keeping original text: {len(main_df) - matches_found}")
    print(f"Manual correction coverage: {matches_found}/{len(main_df)} ({100*matches_found/len(main_df):.1f}%)")
    print(f"Final dataset has single 'Google Books Page Text' column with quotes cleaned where possible")
    
    return output_file

if __name__ == '__main__':
    print("=" * 60)
    print("MERGING MANUAL CORRECTIONS INTO MAIN DATASET")
    print("=" * 60)
    
    final_file = merge_manual_corrections()
    
    print(f"\nMerge complete. Final dataset: {final_file}")
    print("The dataset now contains both quoted and unquoted versions.") 