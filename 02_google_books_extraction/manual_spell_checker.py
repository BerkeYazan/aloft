import pandas as pd
import os
import re
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
import nltk
import logging
from spellchecker import SpellChecker
from nltk import pos_tag, word_tokenize
from PIL import Image
import io

# --- Setup ---
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-much-better-secret-key-for-thesis'

# Download necessary NLTK data for POS tagging
nltk_packages = ['punkt', 'averaged_perceptron_tagger']
for package in nltk_packages:
    try:
        if package == 'punkt':
            nltk.data.find(f'tokenizers/{package}')
        else:
            nltk.data.find(f'taggers/{package}')
    except LookupError:
        logging.info(f"NLTK data package '{package}' not found. Downloading...")
        nltk.download(package, quiet=True)
        logging.info("Download complete.")

# --- Configuration & State Management ---
script_dir = os.path.dirname(__file__)
DATA_FILE = os.path.join(script_dir, 'quotes_cleaned.csv')
CORRECTED_DATA_FILE = os.path.join(script_dir, 'quotes_corrected.csv')
PROGRESS_FILE = os.path.join(script_dir, 'manual_review_progress.txt')
CUSTOM_DICT_FILE = os.path.join(script_dir, 'custom_dictionary.txt')
RECORDED_CORRECTIONS_FILE = os.path.join(script_dir, 'recorded_corrections.json')
SCREENSHOTS_DIR = os.path.join(script_dir, 'Snippet_Screenshots')
COORDINATES_CONFIG = os.path.join(script_dir, 'coordinates.json')

def load_coordinates(config_path):
    """Loads coordinate configurations from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"FATAL: Coordinates config file not found at '{config_path}'. Please run 'define_screenshot_coordinates.py' first.")
        return None
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not parse '{config_path}'. Please check for syntax errors or regenerate it.")
        return None

coordinates = load_coordinates(COORDINATES_CONFIG)

def get_screenshot_category(filename):
    """Determines the category of a screenshot based on its specific filename prefix."""
    date_based_prefixes = [
        '2025', 'Screenshot-2025-07-12', 'Screenshot 2025-07-12',
        'Screenshot-2025-07-13', 'Screenshot 2025-07-13',
        'Screenshot-2025-07-14', 'Screenshot 2025-07-14',
        'Screenshot-2025-07-15', 'Screenshot 2025-07-15',
        'Screenshot-2025-07-16', 'Screenshot 2025-07-16'
    ]
    for prefix in date_based_prefixes:
        if filename.startswith(prefix):
            return 'date_based'
    return 'title_based'


def load_custom_dictionary():
    """Loads a custom dictionary of words to ignore."""
    if not os.path.exists(CUSTOM_DICT_FILE):
        return set()
    with open(CUSTOM_DICT_FILE, 'r', encoding='utf-8') as f:
        return {line.strip().lower() for line in f}

def add_to_custom_dictionary(word):
    """Adds a new word to the custom dictionary."""
    with open(CUSTOM_DICT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{word.lower()}\n")

def get_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            return int(content) if content.isdigit() else 0
    return 0

def save_progress(index):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(index))

def load_recorded_corrections():
    """Loads the dictionary of find/replace corrections."""
    if not os.path.exists(RECORDED_CORRECTIONS_FILE):
        return {}
    with open(RECORDED_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_recorded_corrections(new_corrections):
    """Adds new corrections to the existing dictionary."""
    existing_corrections = load_recorded_corrections()
    existing_corrections.update(new_corrections)
    with open(RECORDED_CORRECTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_corrections, f, indent=4, sort_keys=True)

def apply_recorded_corrections(text, corrections):
    """Applies a dictionary of corrections to a block of text."""
    for error, correction in corrections.items():
        # Use word boundaries (\b) for whole-word-only replacement.
        # This prevents replacing 'he' in 'the'. Case-insensitive.
        text = re.sub(
            r'\b' + re.escape(error) + r'\b',
            correction,
            text,
            flags=re.IGNORECASE
        )
    return text

# --- Core Logic ---
def find_errors_in_text(text, spell_checker, custom_dict):
    """
    Finds potential spelling errors and returns a list of unique error words.
    """
    if not isinstance(text, str):
        return [], text

    # Use NLTK's tokenizer to correctly handle contractions and punctuation.
    tokenized_words = word_tokenize(text)
    # We are only interested in alphabetic words for spell-checking.
    words = [word for word in tokenized_words if word.isalpha()]
    
    # Identify proper nouns using NLTK's POS tagging on the alphabetic words.
    tagged_words = nltk.pos_tag(words)
    proper_nouns = {word.lower() for word, tag in tagged_words if tag in ['NNP', 'NNPS']}

    # Find unknown words, excluding proper nouns and custom dictionary words.
    unknown_words = spell_checker.unknown(words)
    
    potential_errors = {
        word for word in unknown_words 
        if word.lower() not in proper_nouns and word.lower() not in custom_dict
    }

    # Highlight the identified errors in the original text for display
    highlighted_text = text
    if potential_errors:
        # Sort errors by length, longest first, to avoid highlighting issues with substrings
        sorted_errors = sorted(list(potential_errors), key=len, reverse=True)
        error_pattern = r'\b(' + '|'.join(re.escape(error) for error in sorted_errors) + r')\b'
        highlighted_text = re.sub(
            error_pattern,
            r'<mark>\1</mark>',
            highlighted_text,
            flags=re.IGNORECASE
        )
        
    return sorted(list(potential_errors), key=str.lower), highlighted_text


# --- Flask Routes ---

# This route now crops the image on-the-fly before serving it.
@app.route('/screenshot/<path:filename>')
def serve_screenshot(filename):
    if not coordinates:
        return "Coordinates configuration not loaded.", 500

    category = get_screenshot_category(filename)
    boxes = coordinates.get(category)
    if not boxes or 'page_text_box' not in boxes:
        logging.error(f"No coordinates found for category '{category}' in config.")
        # Fallback to sending the full image if something is wrong with the config
        return send_from_directory(SCREENSHOTS_DIR, filename)
    
    try:
        image_path = os.path.join(SCREENSHOTS_DIR, filename)
        img = Image.open(image_path)
        
        # Crop the image to the page text area
        page_text_box = tuple(boxes['page_text_box'])
        cropped_img = img.crop(page_text_box)
        
        # Save the cropped image to a memory buffer
        img_io = io.BytesIO()
        cropped_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except FileNotFoundError:
        return "Screenshot file not found.", 404
    except Exception as e:
        logging.error(f"Error processing screenshot {filename}: {e}")
        return "Error processing image.", 500


@app.route('/')
def index():
    # Determine which file to load: the corrected one if it exists, otherwise the original.
    data_source_path = CORRECTED_DATA_FILE if os.path.exists(CORRECTED_DATA_FILE) else DATA_FILE
    
    try:
        df = pd.read_csv(data_source_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_source_path}")
        return "Error: The source data file could not be found. Please ensure 'quotes_cleaned.csv' is in the correct directory.", 404

    current_index = get_progress()
    corrections = load_recorded_corrections()

    # Load the spell checker and our custom dictionary
    spell = SpellChecker()
    custom_dict = load_custom_dictionary()
    spell.word_frequency.load_words(custom_dict)

    # --- Find the next document that needs review ---
    doc_index = -1
    error_words = []
    
    for i in range(current_index, len(df)):
        row = df.iloc[i]
        
        # Get both text fields
        quote_text = str(row.get('QUOTE', ''))
        page_text = str(row.get('Google Books Page Text', ''))
        
        # Automatically apply all previously recorded global corrections to both fields
        corrected_quote = apply_recorded_corrections(quote_text, corrections)
        corrected_page_text = apply_recorded_corrections(page_text, corrections)

        combined_text_to_review = f"{corrected_quote}\n\n{corrected_page_text}"

        if not combined_text_to_review.strip():
            continue
        
        found_errors, _ = find_errors_in_text(combined_text_to_review, spell, custom_dict)

        if found_errors:
            doc_index = i
            error_words = found_errors
            break  # Found a document with errors, break the loop

    # If we've looped through all remaining files and found no more errors
    if doc_index == -1:
        save_progress(len(df)) # Mark progress as complete
        return render_template('index.html', all_done=True, progress_text="All documents reviewed!")

    # --- We have a document, prepare its data for the template ---
    row = df.iloc[doc_index]
    # Get original texts and apply corrections for display
    original_quote = str(row.get('QUOTE', ''))
    original_page_text = str(row.get('Google Books Page Text', ''))
    
    quote_to_display = apply_recorded_corrections(original_quote, corrections)
    page_text_to_display = apply_recorded_corrections(original_page_text, corrections)

    # Find errors in the combined text to generate a unified error list
    combined_text = f"{quote_to_display}\n\n{page_text_to_display}"
    all_error_words, _ = find_errors_in_text(combined_text, spell, custom_dict)

    # Generate highlighted HTML for each text field separately
    _, quote_highlighted_html = find_errors_in_text(quote_to_display, spell, custom_dict)
    _, page_highlighted_html = find_errors_in_text(page_text_to_display, spell, custom_dict)
    
    # --- Find the associated screenshot filename ---
    screenshot_filename = row.get('Filename') or row.get('filename') or row.get('source') or row.get('source_filename') or None
    if screenshot_filename and not os.path.exists(os.path.join(SCREENSHOTS_DIR, screenshot_filename)):
        logging.warning(f"Screenshot file '{screenshot_filename}' not found in {SCREENSHOTS_DIR}")
        screenshot_filename = None # Don't pass a broken link to the template

    save_progress(doc_index) # Save progress so a refresh loads the same document

    progress_text = f"Reviewing Document {doc_index + 1} of {len(df)} ({len(all_error_words)} potential errors)"
    template_data = {
        'doc_index': doc_index,
        'error_words': all_error_words,
        'quote_context_html': quote_highlighted_html,
        'page_context_html': page_highlighted_html,
        'original_quote': quote_to_display,
        'original_page_text': page_text_to_display,
        'screenshot_filename': screenshot_filename,
        'progress_text': progress_text
    }
    return render_template('index.html', data=template_data)


@app.route('/go_back')
def go_back():
    """Navigate to the previous document with errors."""
    current_index = get_progress()
    
    # Go back to the previous index, but don't go below 0
    previous_index = max(0, current_index - 1)
    
    # If we're already at the beginning, stay at the beginning
    if previous_index == current_index and current_index == 0:
        save_progress(0)
        return redirect(url_for('index'))
    
    # Load the spell checker and custom dictionary to find previous document with errors
    spell = SpellChecker()
    custom_dict = load_custom_dictionary()
    spell.word_frequency.load_words(custom_dict)
    corrections = load_recorded_corrections()
    
    # Determine which file to load
    data_source_path = CORRECTED_DATA_FILE if os.path.exists(CORRECTED_DATA_FILE) else DATA_FILE
    df = pd.read_csv(data_source_path)
    
    # Search backwards from the previous index to find a document with errors
    for i in range(previous_index, -1, -1):
        if i >= len(df):
            continue
            
        row = df.iloc[i]
        quote_text = str(row.get('QUOTE', ''))
        page_text = str(row.get('Google Books Page Text', ''))
        
        # Apply corrections
        corrected_quote = apply_recorded_corrections(quote_text, corrections)
        corrected_page_text = apply_recorded_corrections(page_text, corrections)
        combined_text = f"{corrected_quote}\n\n{corrected_page_text}"
        
        if not combined_text.strip():
            continue
            
        found_errors, _ = find_errors_in_text(combined_text, spell, custom_dict)
        
        if found_errors:
            save_progress(i)
            return redirect(url_for('index'))
    
    # If no previous document with errors found, go to the beginning
    save_progress(0)
    return redirect(url_for('index'))


@app.route('/submit', methods=['POST'])
def submit():
    doc_index = int(request.form['doc_index'])
    edited_quote = request.form.get('edited_quote', '')
    edited_page_text = request.form.get('edited_page_text', '')
    
    # Handle words to be ignored (added to custom dictionary)
    if 'ignore_words' in request.form:
        words_to_ignore = request.form.getlist('ignore_words')
        for word in words_to_ignore:
            if word: # Ensure not to add empty strings
                add_to_custom_dictionary(word)

    # Handle new global corrections to be saved for future use
    new_global_corrections = {}
    for key, value in request.form.items():
        if key.startswith('global_correction-') and value.strip():
            original_word = key.replace('global_correction-', '')
            correction = value.strip()
            # Store keys in lowercase for case-insensitive matching later
            new_global_corrections[original_word.lower()] = correction
    
    if new_global_corrections:
        save_recorded_corrections(new_global_corrections)
            
    # --- Update the DataFrame and save to the new CSV file ---
    # Determine which file to load as the base. If the corrected file exists, use it.
    # Otherwise, start from the original data file.
    data_source_path = CORRECTED_DATA_FILE if os.path.exists(CORRECTED_DATA_FILE) else DATA_FILE
    df = pd.read_csv(data_source_path)
    
    if doc_index < len(df):
        # Save the corrected text for BOTH columns independently.
        df.loc[doc_index, 'QUOTE'] = edited_quote
        df.loc[doc_index, 'Google Books Page Text'] = edited_page_text
        
        # Save the entire dataframe back to the NEW CSV file.
        # This will create the file on the first run and update it on subsequent runs.
        df.to_csv(CORRECTED_DATA_FILE, index=False, encoding='utf-8')
        logging.info(f"Saved corrections for document {doc_index + 1} to '{CORRECTED_DATA_FILE}'")

    # Save progress to the *next* document index. The index() route will find the next one with errors.
    save_progress(doc_index + 1)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    logging.info("--- Professional Manual Spell Checker ---")
    logging.info("Starting the Flask server...")
    logging.info("Open this URL in your web browser: http://127.0.0.1:5001")
    logging.info("Press CTRL+C in this terminal to stop the server.")
    app.run(host='127.0.0.1', port=5001, debug=False) 