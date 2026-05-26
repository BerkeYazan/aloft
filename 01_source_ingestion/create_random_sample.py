import pandas as pd

# Define file paths
INPUT_FILE = 'data/interim/cleaning_data/Data/quotes_with_author_language.csv'
OUTPUT_SAMPLE_FILE = 'data/interim/cleaning_data/Data/random_test_sample.csv'
SAMPLE_SIZE = 50
RANDOM_SEED = 112

def create_sample():
    """
    Reads the main data file, selects a random sample of unique books,
    and saves them to a new CSV.
    """
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Get unique books
    unique_books = df[['TITLE', 'AUTHOR']].drop_duplicates()
    print(f"Found {len(unique_books)} unique books.")

    if len(unique_books) < SAMPLE_SIZE:
        print(f"Warning: Fewer unique books ({len(unique_books)}) than requested sample size ({SAMPLE_SIZE}). Using all unique books.")
        sample_df = unique_books
    else:
        print(f"Taking a random sample of {SAMPLE_SIZE} books with seed {RANDOM_SEED}...")
        sample_df = unique_books.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    # To get the original quotes, we merge back
    final_sample_df = pd.merge(sample_df, df, on=['TITLE', 'AUTHOR'], how='inner').drop_duplicates('TITLE')

    print(f"Saving sample to {OUTPUT_SAMPLE_FILE}...")
    final_sample_df.to_csv(OUTPUT_SAMPLE_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    create_sample() 