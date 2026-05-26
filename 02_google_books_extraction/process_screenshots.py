import pandas as pd
from PIL import Image
import os
import csv
import logging
from tqdm import tqdm
import json
from google.cloud import vision
import io

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

def extract_text_with_google_vision(image_path, coordinates, client):
    """
    Extracts text from defined regions of a screenshot using Google Cloud Vision API.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        logging.error(f"Could not open image {os.path.basename(image_path)}: {e}", exc_info=True)
        return None, None

    category = get_screenshot_category(os.path.basename(image_path))
    boxes = coordinates.get(category)
    if not boxes:
        logging.error(f"No coordinates found for category '{category}' in config file.")
        return None, None

    search_bar_box = tuple(boxes['search_bar_box'])
    page_text_box = tuple(boxes['page_text_box'])

    try:
        # --- Process Search Bar ---
        search_bar_crop = img.crop(search_bar_box)
        with io.BytesIO() as output:
            search_bar_crop.save(output, format="PNG")
            search_bar_content = output.getvalue()
        search_bar_image = vision.Image(content=search_bar_content)
        search_bar_response = client.text_detection(image=search_bar_image)
        search_bar_text = search_bar_response.text_annotations[0].description.strip() if search_bar_response.text_annotations else ""

        # --- Process Page Text ---
        page_text_crop = img.crop(page_text_box)
        with io.BytesIO() as output:
            page_text_crop.save(output, format="PNG")
            page_text_content = output.getvalue()
        page_text_image = vision.Image(content=page_text_content)
        # Use document_text_detection for dense text, as it's more robust.
        page_text_response = client.document_text_detection(image=page_text_image)
        page_text = page_text_response.full_text_annotation.text.strip() if page_text_response.full_text_annotation else ""
        
        return search_bar_text, page_text

    except Exception as e:
        logging.error(f"Error during Google Vision API call for {os.path.basename(image_path)}: {e}", exc_info=True)
        return None, None

def process_all_screenshots(screenshots_dir, output_csv, batch_size, coordinates):
    """
    Processes all PNG images in a directory, extracts text using Google Cloud Vision,
    and saves it to a CSV file.
    """
    # Initialize the Google Vision client once.
    try:
        vision_client = vision.ImageAnnotatorClient()
        logging.info("Google Cloud Vision client initialized successfully.")
    except Exception as e:
        logging.error(f"FATAL: Could not initialize Google Cloud Vision client. Ensure you have authenticated correctly.", exc_info=True)
        return

    processed_files = set()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            if 'Filename' in df_existing.columns:
                processed_files = set(df_existing['Filename'])
                logging.info(f"Found {len(processed_files)} previously processed files.")
        except (pd.errors.EmptyDataError, FileNotFoundError):
            logging.warning("Output CSV not found or is empty. Starting fresh.")
        except Exception as e:
            logging.error(f"Could not read existing CSV file: {e}. Starting fresh.", exc_info=True)

    all_image_files = {f for f in os.listdir(screenshots_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
    files_to_process = sorted(list(all_image_files - processed_files))

    if not files_to_process:
        logging.info("No new images to process. All files are up to date.")
        return

    if not processed_files and (not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Filename", "Title", "Author", "Google Books Page Text"])

    logging.info(f"Processing {len(files_to_process)} new images with Google Cloud Vision...")
    results_batch = []
    
    for filename in tqdm(files_to_process, desc="OCR with Google Vision"):
        image_path = os.path.join(screenshots_dir, filename)
        search_text, page_text = extract_text_with_google_vision(image_path, coordinates, vision_client)

        if search_text is not None and page_text:
            # Simple parsing for title and author from search query
            title = search_text.split('"')[1] if '"' in search_text else 'NA'
            author = search_text.split('inauthor:')[1].strip() if 'inauthor:' in search_text else 'NA'
            results_batch.append([filename, title, author, page_text])
        
        if len(results_batch) >= batch_size:
            # Append batch to CSV
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerows(results_batch)
            results_batch.clear()

    if results_batch:
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerows(results_batch)

    logging.info("\nProcessing finished for all new images.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_file = os.path.join(script_dir, 'ocr_processing.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    config_path = os.path.join(script_dir, 'coordinates.json')
    coordinates = load_coordinates(config_path)
    if coordinates is None:
        exit()

    SCREENSHOTS_DIR = os.path.join(script_dir, 'Snippet_Screenshots')
    # Save to a new file to avoid conflicts with the old Tesseract output
    OUTPUT_CSV = os.path.join(script_dir, 'extracted_text_google_vision.csv')
    BATCH_SIZE = 20

    process_all_screenshots(SCREENSHOTS_DIR, OUTPUT_CSV, BATCH_SIZE, coordinates)