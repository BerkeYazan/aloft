import pandas as pd
import os
from tqdm import tqdm
import re
import ftfy
import difflib
import spacy
import random
import html


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
                                 todesc='Cleaned Page Text')
    full_html = f"<h3>--- CHANGE DETECTED FOR {row_identifier} ---</h3>\n"
    full_html += diff_table
    full_html += "\n<hr>\n"
    return full_html


def remove_drop_cap(text):
    """
    Removes a decorative drop cap from the beginning of a text block.
    This pattern looks for a large initial letter, followed by the rest
    of the word in uppercase, often separated by space.
    e.g., "T OMMY" -> "Tommy" or "T ommy" -> "Tommy"
    """
    # Regex to find a single letter, optional spaces, then the rest of a word in caps.
    # It captures the single letter (group 1) and the rest of the word (group 2).
    match = re.match(r'^\s*([A-Za-z])\s+([A-Z]{2,})\b', text)
    if match:
        first_letter = match.group(1)
        rest_of_word = match.group(2)
        # Reconstruct the word in proper case and prepend it to the rest of the text.
        reconstructed_word = first_letter + rest_of_word.lower()
        # The end of the match is where the matched text (e.g., "T OMMY") finishes.
        text_after_match = text[match.end():]
        return reconstructed_word + text_after_match
    return text


def merge_dialogue_sents(doc):
    """
    Merges short dialogue sentences with their following attribution, which spaCy
    often incorrectly splits. E.g., ["Hello!", "he said."] becomes one sentence.
    """
    sents = list(doc.sents)
    if not sents:
        return []
    merged_sents_spans = []
    i = 0
    while i < len(sents):
        current_sent = sents[i]
        if i + 1 < len(sents):
            next_sent = sents[i+1]
            current_sent_text = current_sent.text.strip()
            next_sent_text = next_sent.text.strip()
            if re.search(r'[.?!](["\'”’]|\s)*$', current_sent_text) and re.match(r'^\s*[a-z]', next_sent_text):
                merged_span = doc[current_sent.start:next_sent.end]
                merged_sents_spans.append(merged_span)
                i += 2
                continue
        merged_sents_spans.append(current_sent)
        i += 1
    return merged_sents_spans


def advanced_clean_ocr_text(text, title, author, nlp):
    """
    Performs a multi-step, robust cleaning of raw OCR page text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # --- Step 1: Perform safe, line-by-line filtering ---
    # This removes page numbers, standalone headers, and author/title lines.
    lines = text.split('\n')
    cleaned_lines = []
    author_lower = str(author).lower()
    title_lower = str(title).lower()

    for line in lines:
        stripped_line = line.strip()
        # Remove lines with no alphabetic characters (e.g., page numbers '468')
        if not re.search(r'[a-zA-Z]', stripped_line):
            continue
        # Remove lines that are formatted like headers (short, all-caps, no punctuation)
        ends_like_sentence = re.search(r'[.?!][\'"”’]?\s*$', stripped_line)
        if len(stripped_line) < 50 and stripped_line.isupper() and not ends_like_sentence:
            continue
        # Remove lines that look like an author or title credit, checking length to be safe
        stripped_lower = stripped_line.lower()
        if author_lower and len(stripped_line) < len(author_lower) * 1.5:
            if difflib.SequenceMatcher(None, stripped_lower, author_lower).ratio() > 0.8:
                continue
        if title_lower and len(stripped_line) < len(title_lower) * 1.5:
            if difflib.SequenceMatcher(None, stripped_lower, title_lower).ratio() > 0.8:
                continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # --- Step 2: Handle drop caps before other substitutions ---
    text = remove_drop_cap(text)
    
    # --- Step 3: Perform all simple, text-wide substitutions ---
    boilerplate = [
        r'Pages?\s+\d+\s+to\s+\d+\s+(?:is|are)\s+not\s+shown\s+in\s+this\s+preview\.?',
        r'Some\s+pages?\s+(?:is|are)\s+omitted\s+from\s+this\s+book\s+preview\.?',
        r'Page\s+\d+\s+is\s+not\s+part\s+of\s+this\s+book\s+preview\.?',
        r'copyrighted\s+material',
        r'Auteursrechtelijk\s+beschermd\s+materiaal',
        r'Sommige\s+pagina[\'\']?s?\s+zijn\s+weggelaten\s+uit\s+dit\s+boekvoorbeeld\.?',
        r'Pagina\s+\d+\s+maakt\s+geen\s+deel\s+uit\s+van\s+dit\s+boekvoorbeeld\.?',
        r'Pagina[\'\']?s?\s+\d+\s+tot\s+\d+\s+(?:wordt|worden)\s+niet\s+weergegeven\s+in\s+dit\s+boekvoorbeeld\.?',
        r'Pagina[\'\']?s?\s+\d+\s+tot\s+en\s+met\s+\d+\s+worden\s+niet\s+getoond\s+in\s+dit\s+voorbeeld\.?',
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    ocr_error_pattern = re.compile(r'[ﬁﬂﬃﬄ■□▲\x0c]')
    text = ocr_error_pattern.sub('', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(\)\.)\s+([A-Z])', r'\1\n\2', text)

    # --- Step 4: Use spaCy for linguistic processing (dialogue merging) ---
    doc = nlp(text)
    sents = merge_dialogue_sents(doc)
    if not sents:
        return ""
    
    # Reconstruct the text from the (potentially merged) sentences.
    # This is much safer than trimming to "valid" sentence boundaries.
    reconstructed_text = "".join(sent.text_with_ws for sent in sents)

    # --- Step 5: Final normalization ---
    final_text = ftfy.fix_text(reconstructed_text)
    final_text = re.sub(r'[ \t\r\f\v]+', ' ', final_text).strip()

    # --- Step 6: Final trim for leading/trailing sentence fragments ---
    # This is a final safety check to remove fragments at the start or end
    # caused by a page break.
    if final_text:
        doc = nlp(final_text)
        sents = list(doc.sents)

        # Check and remove trailing fragment
        if sents:
            last_sent_text = sents[-1].text.strip()
            # A sentence is valid if it ends with punctuation, including inside quotes.
            ends_with_punc = last_sent_text.endswith(('.', '?', '!'))
            ends_with_quote_punc = (
                last_sent_text.endswith(('"', '”', '’', "'")) and
                len(last_sent_text) > 1 and
                last_sent_text[-2] in '.?!'
            )
            if not (ends_with_punc or ends_with_quote_punc):
                sents = sents[:-1]

        # Check and remove leading fragment
        if sents:
            first_sent_text = sents[0].text.strip()
            if not re.search(r'^\s*["\'‘“\[\(-]*[A-Z]', first_sent_text):
                sents = sents[1:]

        if sents:
            final_text = "".join(sent.text_with_ws for sent in sents).strip()
        else:
            final_text = ""
    
    return final_text


def light_clean_text(text):
    """
    A simple normalization function for cleaning quotes.
    """
    if not isinstance(text, str) or len(text.split()) <= 3:
        return ""
    text = ftfy.fix_text(text)
    text = re.sub(r'\s+[-—–]+\s*[\w\s\.]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_dataframe():
    """
    This script loads the merged quotes data, cleans both the Goodreads quotes
    and the OCR page text, and saves the result.
    """
    input_path = 'data/interim/google_books_work/merged_quotes.csv'
    output_path = 'data/interim/google_books_work/quotes_cleaned.csv'
    diff_log_path = 'data/interim/google_books_work/cleaning_diff_log.html'

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print("Dataset loaded.")

    print("\nLoading spaCy NLP model...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded successfully.")
    except OSError:
        print("\n--- spaCy Model Not Found ---")
        print("Please run: python -m spacy download en_core_web_sm")
        return

    print("\n--- Applying lightweight cleaning to 'QUOTE' column ---")
    df['QUOTE_cleaned'] = df['QUOTE'].apply(light_clean_text)

    print("\n--- Applying advanced cleaning to 'Google Books Page Text' column ---")
    all_diffs = []
    df['Google Books Page Text_cleaned'] = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Cleaning page text"):
        raw_page_text = row.get('Google Books Page Text')
        if pd.isna(raw_page_text):
            continue
        cleaned_page_text = advanced_clean_ocr_text(raw_page_text, row['TITLE'], row['AUTHOR'], nlp)
        df.loc[index, 'Google Books Page Text_cleaned'] = cleaned_page_text
        original_no_space = re.sub(r'\s+', '', str(raw_page_text))
        cleaned_no_space = re.sub(r'\s+', '', str(cleaned_page_text))
        if original_no_space != cleaned_no_space:
            if random.randint(1, 10) == 1:
                html_diff = generate_html_diff(str(raw_page_text), str(cleaned_page_text), f"Book: '{row['TITLE']}' (Index: {index})")
                all_diffs.append(html_diff)

    html_header = """
    <!DOCTYPE html><html><head><title>Cleaning Diff Log</title>
    <style>
        body { font-family: sans-serif; line-height: 1.4; } h1 { color: #111; }
        h3 { color: #333; margin-top: 30px; border-top: 2px solid #999; padding-top: 20px;}
        table.diff { font-family: 'Courier New', Courier, monospace; border: 1px solid #a0a0a0; border-collapse: collapse; width: 100%; }
        .diff_header { background-color: #e0e0e0; font-weight: bold; }
        td.diff_header { text-align: right; padding: 2px 4px; } .diff_add { background-color: #ddffdd; }
        .diff_chg { background-color: #ffffcc; } .diff_sub { background-color: #ffdddd; }
    </style></head><body>
    <h1>Log of Text Cleaning Differences</h1>
    <p>Showing a random sample (~10%) of significant changes from OCR page text cleaning.</p>
    """
    html_footer = "</body></html>"
    if all_diffs:
        print(f"\nFound {len(all_diffs)} significant changes to log (sampled at ~10%).")
        with open(diff_log_path, 'w', encoding='utf-8') as f:
            f.write(html_header)
            f.writelines(all_diffs)
            f.write(html_footer)
        print(f"A detailed visual log of changes was saved to '{diff_log_path}'")

    df.drop(columns=['QUOTE', 'Google Books Page Text'], inplace=True)
    df.rename(columns={'QUOTE_cleaned': 'QUOTE', 'Google Books Page Text_cleaned': 'Google Books Page Text'}, inplace=True)
    
    # Drop rows where both text columns are empty after cleaning
    df.dropna(subset=['QUOTE', 'Google Books Page Text'], how='all', inplace=True)
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\nProcessing complete! Cleaned data saved to '{output_path}'")
    except Exception as e:
        print(f"\nError saving the file: {e}")

if __name__ == '__main__':
    clean_dataframe() 