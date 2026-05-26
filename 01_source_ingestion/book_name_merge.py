import pandas as pd
import numpy as np

# Path to the input and output files
file_path = 'data/interim/cleaning_data/Data/quotes_with_publication_dates.csv'
output_path = 'data/interim/cleaning_data/Data/quotes_merged_titles_colon.csv'

try:
    df_to_merge = pd.read_csv(file_path)

    # --- 1. Standardize Author and Title Formatting ---
    print("Standardizing author and title formats...")
    for col in ['AUTHOR', 'TITLE']:
        df_to_merge[col] = df_to_merge[col].fillna('').astype(str).str.strip().str.title()
    df_to_merge.replace({'': np.nan}, inplace=True)
    
    initial_unique_titles = df_to_merge['TITLE'].nunique()
    print(f"Initial number of unique book titles: {initial_unique_titles}")

    # --- 2. Merge "Title: Subtitle" into "Title" ---
    print("Merging titles based on 'Title: Subtitle' pattern...")
    
    # Get a set of all unique, non-null titles for fast lookups
    unique_titles = set(df_to_merge['TITLE'].dropna().unique())
    
    # This dictionary will map the longer title to the shorter base title
    title_map = {}
    
    # Iterate through each unique title to find the pattern
    for title in unique_titles:
        if ':' in title:
            # Extract the base title (the part before the first colon)
            base_title = title.split(':', 1)[0].strip()
            
            # Check if this shorter base title also exists as a standalone title
            if base_title in unique_titles:
                # If it does, we map the longer title to the shorter one
                title_map[title] = base_title
                
    # Apply the mapping to the 'TITLE' column
    df_to_merge['TITLE'] = df_to_merge['TITLE'].replace(title_map)
    
    final_unique_titles = df_to_merge['TITLE'].nunique()
    print(f"\nNumber of unique book titles after targeted merging: {final_unique_titles}")
    print(f"Total reduction of unique titles: {initial_unique_titles - final_unique_titles}")
    print(f"Number of titles merged: {len(title_map)}")

    # --- 3. Save the Cleaned Data ---
    df_to_merge.to_csv(output_path, index=False)
    print(f"\nCleaned data successfully saved to: '{output_path}'")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")