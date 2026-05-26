import pandas as pd
import spacy
import random
import sys
from tqdm import tqdm
from spacy.tokens import Doc


# --- Configuration for Sampling Logic ---
# These parameters control the new, smarter sampling logic.
# They are placed here for easy access and modification.
LOW_THRESHOLD = 0.7   # The snippet can be as short as 70% of the quote's length.
HIGH_THRESHOLD = 1.3  # The snippet can be as long as 130% of the quote's length.
MAX_HIGH_GUARDRAIL = 1.5 # The absolute maximum length allowed (150%) before discarding.


def best_snippet(doc: Doc, target_L: int):
    """
    Return a sentence-aligned snippet from a spaCy Doc object whose word count
    is as close as possible to target_L, subject to bounds.
    This is a deterministic function for reproducibility.
    """
    sents = [sent.text for sent in doc.sents]
    lengths = [len(s.split()) for s in sents]
    
    if not sents:
        return None

    best = None
    best_diff = float("inf")
    lower_bound = int(LOW_THRESHOLD * target_L)
    upper_bound = int(HIGH_THRESHOLD * target_L)
    absolute_max = int(MAX_HIGH_GUARDRAIL * target_L) + 1 # Add 1 for safety with rounding
    
    # --- Primary Search: Find best contiguous block within thresholds ---
    for i in range(len(sents)):
        current_len = 0
        for j in range(i, len(sents)):
            current_len += lengths[j]
            
            # Check if the current block is a candidate
            if lower_bound <= current_len <= upper_bound:
                diff = abs(current_len - target_L)
                if diff < best_diff:
                    best = " ".join(sents[i:j+1])
                    best_diff = diff
            
            # Optimization: If we are already way over the absolute max, stop building this block
            if current_len > absolute_max + 50:
                 break
        
        # Early exit if we find a perfect match
        if best_diff == 0:
            break
    
    # --- Fallback: If no block was found, find the single best sentence ---
    if best is None:
        # Check if there are any sentences to process
        if not lengths:
            return None
        diffs = [abs(l - target_L) for l in lengths]
        idx = int(min(range(len(diffs)), key=diffs.__getitem__))
        best = sents[idx]
    
    # --- Final Guard Rail: Discard if the final result is too long OR too short ---
    final_len = len(best.split())
    # The snippet must not be excessively long
    if final_len > absolute_max:
        return None
    # After all fallbacks, it must still be reasonably close (at least half the quote's length)
    # This catches the cases where a very long quote gets a tiny fallback snippet.
    if final_len < target_L * 0.5:
         return None
        
    return best

def main():
    # Load the dataset
    input_file_path = 'data/interim/google_books_work/quotes_final_cleaned.csv'
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {input_file_path}")
        sys.exit(1)

    # Load spacy model
    print("Loading spaCy model 'en_core_web_sm'...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model 'en_core_web_sm'...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")

    # --- Optimized Processing with nlp.pipe ---
    print("Processing text with spaCy using nlp.pipe for efficiency...")
    
    # Get texts, ensuring they are strings to prevent errors with spaCy
    page_texts = df["Google Books Page Text"].fillna('').astype(str).tolist()

    # Process texts in batches using multiple cores for a significant speedup
    docs = list(tqdm(nlp.pipe(page_texts, n_process=-1), total=len(page_texts), desc="SpaCy Processing"))

    print("\nGenerating matched snippets...")
    snippets = []
    # Loop through the dataframe and the pre-processed docs together
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating Snippets"):
        doc = docs[i]
        quote_text = row.get("QUOTE")
        # Also check the original page text column for missing values
        page_text = row.get("Google Books Page Text")

        if pd.isna(quote_text) or pd.isna(page_text):
            snippets.append(None)
            continue
            
        quote_len = len(str(quote_text).split())
        
        snippet = best_snippet(doc, quote_len)
        snippets.append(snippet)
        
    df['MATCHED_SNIPPET'] = snippets

    # --- Final Analysis and Saving ---
    # Drop rows where a snippet couldn't be generated (outliers)
    final_df = df.dropna(subset=['MATCHED_SNIPPET']).copy()
    
    original_count = len(df)
    final_count = len(final_df)
    discarded_count = original_count - final_count

    print("\n--- Processing Complete ---")
    print(f"Successfully generated snippets for {final_count} out of {original_count} quotes ({final_count/original_count:.2%}).")
    print(f"Discarded {discarded_count} quotes ({discarded_count/original_count:.2%}) that did not meet the length criteria.")
    
    # Save the full dataset with the new, normalized snippets
    output_path = 'data/interim/google_books_work/quotes_pages_snippets_cleaned.csv'
    print(f"\nSaving the normalized dataset to:\n{output_path}")
    final_df.to_csv(output_path, index=False)
    print("\nScript finished successfully.")

if __name__ == '__main__':
    main() 