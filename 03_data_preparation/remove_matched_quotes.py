import pandas as pd
import difflib
import re
from tqdm import tqdm

def generate_html_diff(original_text, cleaned_text, row_identifier):
    """
    Generates a rich HTML string that visually highlights the differences
    between the original and cleaned text using a side-by-side table.
    """
    fromlines = original_text.splitlines()
    tolines = cleaned_text.splitlines()
    diff = difflib.HtmlDiff(wrapcolumn=80)
    diff_table = diff.make_table(fromlines, tolines,
                                 fromdesc='Original Page Text',
                                 todesc='Page Text After Quote Removal')
    full_html = f"<h3>--- CHANGE DETECTED FOR {row_identifier} ---</h3>\n"
    full_html += diff_table
    full_html += "\n<hr>\n"
    return full_html

def remove_fuzzy_substring(main_string, sub_string):
    """
    Finds and removes a fuzzy substring from a main string.
    This version uses get_matching_blocks to handle multi-part matches
    and is more robust to newlines and minor OCR differences.
    """
    if not isinstance(main_string, str) or not isinstance(sub_string, str):
        return main_string

    # SequenceMatcher finds all matching blocks between the two strings.
    s = difflib.SequenceMatcher(None, main_string, sub_string, autojunk=False)
    
    # We get all blocks and filter out small, likely spurious matches.
    # The minimum match length can be adjusted.
    min_match_len = 4
    matching_blocks = [m for m in s.get_matching_blocks() if m.size >= min_match_len]

    if not matching_blocks:
        return main_string

    # To avoid removing disconnected, spurious matches, we check if the main
    # blocks are contiguous in the substring.
    first_block_in_sub = matching_blocks[0].b
    last_block_in_sub = matching_blocks[-1].b + matching_blocks[-1].size
    span_in_sub = last_block_in_sub - first_block_in_sub
    
    total_match_in_span = sum(m.size for m in matching_blocks)

    # If the matched characters make up a high percentage of the substring span,
    # we proceed with the removal. This validates that we found a genuine quote.
    if span_in_sub == 0 or total_match_in_span / span_in_sub < 0.8:
        return main_string

    # --- NEW: Check for excessive distance between matches in the main string ---
    # This prevents incorrectly stitching together distant fragments.
    max_allowed_gap = len(sub_string) // 2  # Gap shouldn't be more than half the quote length
    for i in range(len(matching_blocks) - 1):
        current_block_end = matching_blocks[i].a + matching_blocks[i].size
        next_block_start = matching_blocks[i+1].a
        gap = next_block_start - current_block_end
        if gap > max_allowed_gap:
            return main_string # Abort removal if gap is too large

    # We "paint" the indices of the main_string that should be removed.
    indices_to_remove = set()
    for m in matching_blocks:
        for i in range(m.size):
            indices_to_remove.add(m.a + i)

    # Reconstruct the string without the characters at the marked indices.
    result_chars = [char for i, char in enumerate(main_string) if i not in indices_to_remove]
    
    # Clean up any leftover whitespace from the removal.
    cleaned_result = re.sub(r'  +', ' ', "".join(result_chars))
    cleaned_result = re.sub(r'( \n)', '\n', cleaned_result) # space before newline
    cleaned_result = re.sub(r'(\n )', '\n', cleaned_result) # space after newline
    
    return cleaned_result.strip()

def clean_remaining_artifacts(text):
    """
    Cleans up leftover artifacts after quote removal, such as empty lines
    or lines with only punctuation.
    """
    if not isinstance(text, str):
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep a line only if it contains at least one letter or number
        if re.search(r'[a-zA-Z0-9]', stripped):
            cleaned_lines.append(stripped)
            
    return "\n".join(cleaned_lines)

def process_matches(input_path, output_path, diff_log_path):
    """
    Loads the fuzzy matches, removes the matched quotes from the page text,
    and saves the results and a diff log.
    """
    print(f"Loading matches from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return

    df.dropna(subset=['Google Books Page Text', 'matched_quotes_list'], inplace=True)

    all_diffs = []
    results = []
    
    print("Removing matched quotes from page text...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        page_text = row['Google Books Page Text']
        # Split the combined string of quotes back into a list
        quotes_to_remove = row['matched_quotes_list'].split('|||')
        
        unquoted_text = page_text
        # Iteratively remove each quote from the page text
        for quote in quotes_to_remove:
            unquoted_text = remove_fuzzy_substring(unquoted_text, quote)
        
        # Apply the final artifact cleaning after all removals
        final_cleaned_text = clean_remaining_artifacts(unquoted_text)

        new_row = row.to_dict()
        new_row['Google Books Page Text_unquoted'] = final_cleaned_text
        results.append(new_row)
        
        # Generate a diff to log the change, but only if text actually changed
        if page_text != final_cleaned_text:
            html_diff = generate_html_diff(page_text, final_cleaned_text, f"Book: '{row['TITLE']}' (Index: {index})")
            all_diffs.append(html_diff)

    # Create the output dataframe
    result_df = pd.DataFrame(results)
    
    # Save the new CSV
    try:
        result_df.to_csv(output_path, index=False)
        print(f"\nProcessing complete! Unquoted data saved to '{output_path}'")
    except Exception as e:
        print(f"\nError saving the data file: {e}")

    # Save the HTML diff log
    html_header = """
    <!DOCTYPE html><html><head><title>Quote Removal Diff Log</title>
    <style>
        body { font-family: sans-serif; line-height: 1.4; } h1 { color: #111; }
        h3 { color: #333; margin-top: 30px; border-top: 2px solid #999; padding-top: 20px;}
        table.diff { font-family: 'Courier New', Courier, monospace; border: 1px solid #a0a0a0; border-collapse: collapse; width: 100%; }
        .diff_header { background-color: #e0e0e0; font-weight: bold; }
        td.diff_header { text-align: right; padding: 2px 4px; } .diff_add { background-color: #ddffdd; }
        .diff_chg { background-color: #ffffcc; } .diff_sub { background-color: #ffdddd; }
    </style></head><body>
    <h1>Log of Quote Removals from Page Text</h1>
    <p>This log shows the 'matched_quote' being removed from the 'Google Books Page Text'.</p>
    """
    html_footer = "</body></html>"
    if all_diffs:
        print(f"\nFound {len(all_diffs)} removals to log.")
        with open(diff_log_path, 'w', encoding='utf-8') as f:
            f.write(html_header)
            f.writelines(all_diffs)
            f.write(html_footer)
        print(f"A detailed visual log of removals was saved to '{diff_log_path}'")
    else:
        print("No quotes were removed.")

if __name__ == '__main__':
    INPUT_FILE = "data/interim/google_books_work/cross_file_fuzzy_matches.csv"
    OUTPUT_FILE = "data/interim/google_books_work/unquoted_google_books_text.csv"
    DIFF_LOG_FILE = "data/interim/google_books_work/quote_removal_diff_log.html"
    
    process_matches(INPUT_FILE, OUTPUT_FILE, DIFF_LOG_FILE) 