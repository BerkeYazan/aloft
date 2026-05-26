import pandas as pd
import os
import logging
from flask import Flask, render_template, request, redirect, url_for
import difflib
import html
import re

# --- Setup ---
logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'a-super-secret-key-for-cleanup'

# --- Configuration & State Management ---
script_dir = os.path.dirname(__file__)
DATA_FILE = os.path.join(script_dir, 'unquoted_google_books_text.csv')
CORRECTED_DATA_FILE = os.path.join(script_dir, 'unquoted_text_manual_review.csv')
PROGRESS_FILE = os.path.join(script_dir, 'cleanup_review_progress.txt')

def get_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            return int(content) if content.isdigit() else 0
    return 0

def save_progress(index):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(index))

def create_final_editable_html(original, cleaned):
    """
    Creates a single HTML string with corrected highlighting.
    - Removed text is struck through and red.
    - Only the single word immediately before and after substantial deletion blocks is gold.
    - Ignores whitespace-only changes (line breaks, spaces, etc.)
    """
    # Handle NaN/None values by converting to empty strings
    if pd.isna(original) or original is None:
        original = ""
    if pd.isna(cleaned) or cleaned is None:
        cleaned = ""
    
    # Ensure we have strings
    original = str(original)
    cleaned = str(cleaned)
    
    original_tokens = re.split(r'(\s+)', original)
    cleaned_tokens = re.split(r'(\s+)', cleaned)
    s = difflib.SequenceMatcher(None, original_tokens, cleaned_tokens, autojunk=False)
    
    # Find substantial deletion blocks (ignore whitespace-only changes)
    deletion_blocks = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'delete' or tag == 'replace':
            # Check if this deletion contains any non-whitespace content
            deleted_tokens = original_tokens[i1:i2]
            has_content = any(token.strip() for token in deleted_tokens)
            
            if has_content:
                deletion_blocks.append((i1, i2))
    
    # Find words to highlight (one before and one after each substantial deletion block)
    gold_indices = set()
    for start_idx, end_idx in deletion_blocks:
        # Find the word immediately before this deletion block
        for i in range(start_idx - 1, -1, -1):
            if original_tokens[i].strip():  # Found a non-whitespace token
                gold_indices.add(i)
                break
        
        # Find the word immediately after this deletion block
        for i in range(end_idx, len(original_tokens)):
            if original_tokens[i].strip():  # Found a non-whitespace token
                gold_indices.add(i)
                break
    
    # Mark deleted tokens (only from substantial deletion blocks)
    deleted_indices = set()
    for start_idx, end_idx in deletion_blocks:
        for i in range(start_idx, end_idx):
            deleted_indices.add(i)

    # Build the final HTML string
    output = []
    for i, token in enumerate(original_tokens):
        escaped_token = html.escape(token)
        if i in deleted_indices:
            output.append(f'<del style="background-color: #ffdddd;">{escaped_token}</del>')
        elif i in gold_indices:
            output.append(f'<span style="background-color: gold;">{escaped_token}</span>')
        else:
            output.append(escaped_token)
            
    return "".join(output)

@app.route('/')
def index():
    data_source_path = CORRECTED_DATA_FILE if os.path.exists(CORRECTED_DATA_FILE) else DATA_FILE
    try:
        df = pd.read_csv(data_source_path)
    except FileNotFoundError:
        return "Error: Source data file not found.", 404
    
    current_index = get_progress()
    review_row_index = -1
    for i in range(current_index, len(df)):
        row = df.iloc[i]
        if str(row.get('Google Books Page Text', '')) != str(row.get('Google Books Page Text_unquoted', '')):
            review_row_index = i
            break
            
    if review_row_index == -1:
        save_progress(len(df))
        return render_template('quote_cleanup_template.html', all_done=True, progress_text="All documents reviewed!")

    row_data = df.iloc[review_row_index].to_dict()
    save_progress(review_row_index)
    
    total_to_review = df[df['Google Books Page Text'] != df['Google Books Page Text_unquoted']].shape[0]
    reviewed_count = df[(df['Google Books Page Text'] != df['Google Books Page Text_unquoted']) & (df.index < review_row_index)].shape[0]
    progress_text = f"Reviewing Item {reviewed_count + 1} of {total_to_review}"
    
    original_text = row_data.get('Google Books Page Text', '')
    cleaned_text = row_data.get('Google Books Page Text_unquoted', '')
    
    # Handle NaN values from pandas
    if pd.isna(original_text):
        original_text = ''
    if pd.isna(cleaned_text):
        cleaned_text = ''
        
    editable_html = create_final_editable_html(original_text, cleaned_text)

    template_data = {
        'doc_index': review_row_index, 'title': row_data.get('TITLE', 'N/A'),
        'author': row_data.get('AUTHOR', 'N/A'), 'editable_html': editable_html,
        'progress_text': progress_text
    }
    return render_template('quote_cleanup_template.html', data=template_data)

@app.route('/submit', methods=['POST'])
def submit():
    doc_index = int(request.form['doc_index'])
    edited_text = request.form.get('edited_text', '')
    
    data_source_path = CORRECTED_DATA_FILE if os.path.exists(CORRECTED_DATA_FILE) else DATA_FILE
    df = pd.read_csv(data_source_path)
    
    if doc_index < len(df):
        df.loc[doc_index, 'Google Books Page Text_unquoted'] = edited_text
        df.to_csv(CORRECTED_DATA_FILE, index=False, encoding='utf-8')
        logging.info(f"Saved manual correction for document index {doc_index}")

    save_progress(doc_index + 1)
    return redirect(url_for('index'))

if __name__ == '__main__':
    logging.info("--- Manual Quote Cleanup Tool ---")
    logging.info("Starting the Flask server...")
    logging.info("Open this URL in your web browser: http://127.0.0.1:5002")
    logging.info("Press CTRL+C in this terminal to stop the server.")
    app.run(host='127.0.0.1', port=5002, debug=False) 