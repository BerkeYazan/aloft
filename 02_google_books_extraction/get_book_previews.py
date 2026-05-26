import pandas as pd
import time
import requests
import logging
import os
import ast
import html
import re
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
INPUT_FILE = 'data/interim/google_books_work/random_book_likes_weighted.csv'
OUTPUT_FILE = 'data/interim/google_books_work/google_books_sample_output.csv'
PROGRESS_FILE = 'data/interim/google_books_work/link_fetching_progress.txt'
USER_AGENT = "MyThesisProject/1.0"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1 # seconds
BATCH_SIZE = 10 # Save progress after every 10 books

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler("google_books_processing.log"),
                              logging.StreamHandler()])

def get_start_index():
    """Reads the last processed index from the progress file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    # Return the index of the *next* book to process
                    return int(content) + 1
        except (ValueError, IOError) as e:
            logging.warning(f"[WARNING] Could not read progress file '{PROGRESS_FILE}': {e}. Starting from the beginning.")
    return 0

def save_progress(index):
    """Saves the last processed index to the progress file."""
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, 'w') as f:
            f.write(str(index))
    except IOError as e:
        logging.error(f"    [ERROR] Could not save progress to '{PROGRESS_FILE}': {e}")

def fetch_google_books_data(session, title, author, api_key):
    """Fetches data from Google Books API with retries."""
    base_url = "https://www.googleapis.com/books/v1/volumes"
    query_parts = []
    
    if pd.notna(title) and title.strip():
        query_parts.append(f'intitle:"{title}"')
    if pd.notna(author) and author.strip():
        query_parts.append(f'inauthor:"{author}"')
    
    if not query_parts:
        logging.warning("No title or author provided for search.")
        return None

    query = "+".join(query_parts)
    params = {
        "q": query,
        "maxResults": 1,
        "fields": "totalItems,items(volumeInfo(categories,previewLink,pageCount),accessInfo(viewability))",
        "key": api_key
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"    [WARNING] API request failed: {e}. Retrying ({attempt + 1}/{MAX_RETRIES})...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF * (2 ** attempt))
            else:
                logging.error(f"    [ERROR] All API retries failed for '{title}'.")
                return None

def main():
    """
    Main function to load book data, sample it, fetch data from Google Books API,
    and save the results to a CSV file.
    """
    logging.info("\n============================================================")
    logging.info("           STARTING GOOGLE BOOKS PROCESSING RUN")
    logging.info("============================================================")

    if not GOOGLE_BOOKS_API_KEY:
        logging.error("[ERROR] GOOGLE_BOOKS_API_KEY environment variable not set. Exiting.")
        return

    try:
        df = pd.read_csv(INPUT_FILE, sep=';')
        logging.info(f"[INFO] Loaded {len(df)} total books from '{INPUT_FILE}'.")
    except FileNotFoundError:
        logging.error(f"[ERROR] Input file not found: '{INPUT_FILE}'. Exiting.")
        return
    except Exception as e:
        logging.error(f"[ERROR] Error loading '{INPUT_FILE}': {e}. Exiting.")
        return

    # Determine where to start processing from
    start_index = get_start_index()
    if start_index > 0:
        logging.info(f"[INFO] Resuming from book at index {start_index}.")

    df_to_process = df.iloc[start_index:]
    
    if df_to_process.empty:
        logging.info("[INFO] All books from input file have been processed. To re-run, delete the progress file.")
        return
    
    logging.info(f"[INFO] {len(df_to_process)} books remaining to process.")
    logging.info("------------------------------------------------------------")

    # Determine if the output CSV already exists and has a header
    should_append = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0

    results = []
    total_processed_in_run = 0
    cache = {}
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for index, row in df_to_process.iterrows():
            title = row.get('TITLE')
            author = row.get('AUTHOR')
            
            logging.info(f"\n--- Processing book {index}/{len(df)-1}: '{title}' ---")
            
            cache_key = (title, author)
            
            if cache_key in cache:
                logging.info("    [INFO] Using cached result.")
                data = cache[cache_key]
            else:
                logging.info("    Searching Google Books API...")
                data = fetch_google_books_data(session, title, author, GOOGLE_BOOKS_API_KEY)
                cache[cache_key] = data
                time.sleep(1) # Be nice to the API

            if data and data.get("totalItems", 0) > 0 and "items" in data:
                item = data["items"][0]
                volume_info = item.get("volumeInfo", {})
                access_info = item.get("accessInfo", {})
                
                preview_link = volume_info.get("previewLink")
                viewability = access_info.get("viewability") if access_info else None

                if preview_link and viewability in ['PARTIAL', 'PAGES']:
                    page_count = volume_info.get("pageCount")
                    
                    page_specific_link = preview_link.replace('&printsec=frontcover', '')

                    if page_count and page_count > 1:
                        random_page = random.randint(1, page_count)
                        page_specific_link = f"{page_specific_link}&pg=PA{random_page}"
                        logging.info(f"    [SUCCESS] Preview found. Generated link to page {random_page} of {page_count}.")
                    else:
                        logging.warning(f"    [WARNING] No page count available, using default preview link.")

                    genres = volume_info.get("categories", [])
                    
                    result = row.to_dict()
                    result.update({
                        "genres": ", ".join(genres) if genres else None,
                        "preview_link": page_specific_link
                    })
                    results.append(result)

                    # --- Batch Saving Logic ---
                    if len(results) >= BATCH_SIZE:
                        try:
                            results_df = pd.DataFrame(results)
                            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
                            results_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not should_append)
                            
                            logging.info("------------------------------------------------------------")
                            logging.info(f"[SUCCESS] Saved a batch of {len(results_df)} results.")
                            logging.info("------------------------------------------------------------")
                            
                            total_processed_in_run += len(results)
                            should_append = True # From now on, we always append
                            results = [] # Reset the batch
                        except Exception as e:
                            logging.error(f"    [ERROR] Failed to save batch results: {e}")

                else:
                    logging.warning(f"    [FAILURE] No preview available (Viewability: {viewability}). Skipping.")
            else:
                logging.warning("    [FAILURE] No data found for this book.")
            
            # Save progress after every book attempt
            save_progress(index)

    # --- Save any remaining results after the loop ---
    if results:
        try:
            results_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            results_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not should_append)
            total_processed_in_run += len(results)
            logging.info("------------------------------------------------------------")
            logging.info(f"[SUCCESS] Saved the final batch of {len(results)} results.")
        except Exception as e:
            logging.error(f"    [ERROR] Failed to save final batch results: {e}")
    
    logging.info("\n============================================================")
    logging.info("              GOOGLE BOOKS PROCESSING FINISHED")
    logging.info("============================================================")
    logging.info(f"[INFO] Total books successfully processed in this run: {total_processed_in_run}")
    logging.info("============================================================\n")


if __name__ == "__main__":
    main()

