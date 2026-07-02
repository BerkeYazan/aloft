import pandas as pd
import requests
import time
from thefuzz import fuzz
from tqdm import tqdm
import re
import logging
import os
import argparse
from typing import List, Tuple, Dict, Set, Optional, Any
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# --- Setup Logging ---
# A more detailed format for better debugging
log_format = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
# Use rotating file handler to prevent log file from growing too large
log_handler = RotatingFileHandler(
    "book_date_fetching.log", 
    maxBytes=50*1024*1024,  # 50MB max file size
    backupCount=5,  # Keep 5 backup files
    mode='w'
)
log_handler.setFormatter(logging.Formatter(log_format))

logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    handlers=[log_handler, logging.StreamHandler()])

# --- Configuration ---
WIKIDATA_API_URL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
# Wikidata asks API clients to identify themselves with contact details.
# Set WIKIDATA_CONTACT_EMAIL in .env; the placeholder default also works.
CONTACT_EMAIL = os.getenv("WIKIDATA_CONTACT_EMAIL", "your-email@example.com")
USER_AGENT = f"ALOFT-BookDateFetcher/1.1 (Contact: {CONTACT_EMAIL})"
HEADERS = {'Accept': 'application/sparql-results+json', 'User-Agent': USER_AGENT}
MAX_RETRIES = 3
INITIAL_BACKOFF = 2
API_DELAY_S = 0.3  # Optimized for Wikidata guidelines (200 req/min = 0.3s)
MIN_DELAY_S = 0.2  # Minimum delay between requests
MAX_DELAY_S = 10.0  # Maximum delay for adaptive rate limiting

# Global variable to track consecutive failures for adaptive rate limiting
consecutive_failures = 0

# Keywords to identify book-like items in Wikidata descriptions
BOOK_KEYWORDS = [
    'novel', 'book', 'play', 'poem', 'short story', 'non-fiction', 'fiction',
    'collection', 'novella', 'anthology', 'publication', 'series of novels',
    'literary work', 'treatise', 'memoir', 'autobiography', 'biography', 'written work'
]

# [NEW] More reliable keywords for checking an entity's 'instance of' (P31) property.
BOOK_INSTANCE_KEYWORDS = [
    'book', 'literary work', 'novel', 'novella', 'short story', 'poem', 'play', 'screenplay',
    'written work', 'book series', 'anthology', 'collection of short stories', 'treatise',
    'memoir', 'autobiography', 'biography', 'prose', 'graphic novel', 'comedy', 'tragedy',
    'pastoral', 'libretto', 'script', 'non-fiction', 'fiction', 'essay', 'epic poetry'
]

# --- Robust API Functions ---

def make_api_request_with_retry(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Optional[requests.Response]:
    """Makes an HTTP GET request with retries, exponential backoff, and adaptive rate limiting."""
    global consecutive_failures
    
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)
    
    # Apply adaptive delay based on recent failures
    if consecutive_failures > 0:
        adaptive_delay = min(API_DELAY_S * (1.5 ** consecutive_failures), MAX_DELAY_S)
        logging.info(f"Adaptive delay: {adaptive_delay:.1f}s due to {consecutive_failures} consecutive failures")
        time.sleep(adaptive_delay)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429:  # Rate limit
                retry_after = int(response.headers.get("Retry-After", INITIAL_BACKOFF * (2 ** attempt)))
                logging.warning(f"Rate limit hit (429). Retrying in {retry_after}s...")
                consecutive_failures += 1
                time.sleep(retry_after)
                continue
            elif response.status_code == 503:  # Service unavailable
                logging.warning(f"Service unavailable (503). Retrying in {INITIAL_BACKOFF * (2 ** attempt)}s...")
                consecutive_failures += 1
                time.sleep(INITIAL_BACKOFF * (2 ** attempt))
                continue
            
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            
            # Reset consecutive failures on success
            consecutive_failures = max(0, consecutive_failures - 1)
            return response
            
        except requests.exceptions.Timeout:
            logging.warning(f"Request timeout. (Attempt {attempt + 1}/{MAX_RETRIES})")
            consecutive_failures += 1
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF * (2 ** attempt))
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed: {e}. (Attempt {attempt + 1}/{MAX_RETRIES})")
            consecutive_failures += 1
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF * (2 ** attempt))
            else:
                logging.error(f"All {MAX_RETRIES} retries failed for URL: {url} with params: {params}")
                return None
    
    return None

def get_data_for_qids(qids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    [REFACTORED V5 - Final] Fetches authors, dates, and instance types using three
    separate, simple queries to ensure maximum reliability and avoid SPARQL UNION issues.
    """
    if not qids:
        return {}

    # Initialize results structure
    final_results = {qid: {'authors': set(), 'dates': set(), 'instances': set()} for qid in qids}
    values_clause = " ".join([f"wd:{qid}" for qid in qids])

    # --- Query 1: Get Authors ---
    query_authors = f"""
    SELECT ?item ?authorLabel WHERE {{
      VALUES ?item {{ {values_clause} }}
      ?item wdt:P50 ?author.
      ?author rdfs:label ?authorLabel. FILTER(LANG(?authorLabel) = "en")
    }}
    """
    try:
        response = make_api_request_with_retry(WIKIDATA_API_URL, params={'query': query_authors}, headers=HEADERS)
        if response:
            for item in response.json().get('results', {}).get('bindings', []):
                qid = item.get('item', {}).get('value', '').split('/')[-1]
                if qid in final_results and 'authorLabel' in item:
                    final_results[qid]['authors'].add(item['authorLabel']['value'])
    except (Exception) as e:
        logging.error(f"Error fetching authors: {e}")

    # --- Query 2: Get Dates ---
    query_dates = f"""
    SELECT ?item ?date_val WHERE {{
      VALUES ?item {{ {values_clause} }}
      {{ ?item wdt:P577 ?date_val. }} UNION {{ ?item wdt:P580 ?date_val. }} UNION {{ ?item wdt:P571 ?date_val. }}
    }}
    """
    try:
        response = make_api_request_with_retry(WIKIDATA_API_URL, params={'query': query_dates}, headers=HEADERS)
        if response:
            for item in response.json().get('results', {}).get('bindings', []):
                qid = item.get('item', {}).get('value', '').split('/')[-1]
                if qid in final_results and 'date_val' in item:
                    # Parse the year from the raw date string in Python for robustness
                    raw_date = item['date_val']['value']
                    match = re.search(r'^\+?(\d{4})', raw_date)
                    if match:
                        final_results[qid]['dates'].add(int(match.group(1)))
    except (Exception) as e:
        logging.error(f"Error fetching dates: {e}")

    # --- Query 3: Get Instance Types ---
    query_instances = f"""
    SELECT ?item ?instance_of_label WHERE {{
      VALUES ?item {{ {values_clause} }}
      ?item wdt:P31 ?instance_of.
      ?instance_of rdfs:label ?instance_of_label. FILTER(LANG(?instance_of_label) = "en")
    }}
    """
    try:
        response = make_api_request_with_retry(WIKIDATA_API_URL, params={'query': query_instances}, headers=HEADERS)
        if response:
            for item in response.json().get('results', {}).get('bindings', []):
                qid = item.get('item', {}).get('value', '').split('/')[-1]
                if qid in final_results and 'instance_of_label' in item:
                    final_results[qid]['instances'].add(item['instance_of_label']['value'].lower())
    except (Exception) as e:
        logging.error(f"Error fetching instances: {e}")
    
    # Finalize the results: find the minimum date and convert sets to lists
    for qid in final_results:
        values = final_results[qid]
        earliest_date = min(values['dates']) if values['dates'] else None
        final_results[qid] = {
            'authors': list(values['authors']),
            'date': str(earliest_date) if earliest_date else None,
            'instances': list(values['instances'])
        }
    return final_results

def find_best_book_match(book_title: str, author_name: str) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
    """
    [REFACTORED V6 - Final] Finds the best Wikidata match with a multi-stage search.
    - Handles special cases like '1984'.
    - Uses fuzz.token_set_ratio for more robust title matching against subtitles.
    - Tries multiple query variations, stopping early on a high-confidence match.
    - Uses an expanded list of 'instance of' (P31) keywords for higher accuracy.
    """
    # --- Step 1: Prepare Search Queries ---
    
    # Handle special, known cases first.
    if book_title == '1984':
        search_titles = ['Nineteen Eighty-Four', book_title]
    else:
        search_titles = [book_title]
    
    # Also add subtitle variations
    if ':' in book_title:
        parts = book_title.split(':', 1)
        if parts[0].strip(): search_titles.append(parts[0].strip())
        if len(parts) > 1 and parts[1].strip(): search_titles.append(parts[1].strip())
        
    # Build a list of query variations in order of priority.
    # We will iterate through these and stop at the first high-confidence match.
    search_permutations = []
    for title in list(dict.fromkeys(search_titles)): # Deduplicate titles
        # Store as a tuple: (title_to_search, author_to_search)
        search_permutations.append((f"{title} {author_name}", title))
        search_permutations.append((title, title))
    
    all_candidates = []
    seen_qids = set()

    # --- Step 2: Execute Searches and Find a High-Confidence Match ---
    for query, query_title in search_permutations: # Unpack the tuple
        params = {'action': 'wbsearchentities', 'format': 'json', 'language': 'en', 'type': 'item', 'search': query, 'limit': 9}
        response = make_api_request_with_retry(WIKIDATA_SEARCH_URL, params=params)
        time.sleep(API_DELAY_S)
        if not response: continue

        try:
            search_results = response.json().get('search', [])
            if not search_results: continue

            # --- Step 2a: Quick Scan for a High-Confidence Match ---
            qids_to_fetch = [res['id'] for res in search_results if res.get('id') not in seen_qids]
            if not qids_to_fetch: continue
            
            bulk_data = get_data_for_qids(qids_to_fetch)

            # Analyze these new candidates
            current_candidates = []
            author_lower = author_name.lower()
            for result in search_results:
                qid = result.get('id')
                if qid in seen_qids: continue # Skip if already processed from a previous permutation
                seen_qids.add(qid)

                label = result.get('label', '')
                desc = result.get('description', '').lower()
                entity_data = bulk_data.get(qid, {'authors': [], 'date': None, 'instances': []})
                
                # [REVISED SCORING V3] Score title against the specific query title variation
                title_score = fuzz.token_set_ratio(query_title.lower(), label.lower())
                instance_score = 150 if any(inst in BOOK_INSTANCE_KEYWORDS for inst in entity_data['instances']) else 0
                
                # Stricter author matching.
                author_score = 0
                if entity_data['authors']:
                    author_scores = [fuzz.ratio(author_lower, api_author.lower()) for api_author in entity_data['authors']]
                    best_author_score = max(author_scores) if author_scores else 0
                    if best_author_score < 75:
                        author_score = -500 # Heavy penalty for author mismatch
                    else:
                        author_score = best_author_score
                # Fallback to description/label check only if no structured author is found
                else:
                    if f"by {author_lower}" in desc: author_score = max(author_score, 90)
                    if author_lower in label.lower(): author_score = max(author_score, 80)

                desc_score = 50 if instance_score == 0 and any(kw in desc for kw in BOOK_KEYWORDS) else 0
                
                # Heavy penalty if the item is an 'edition' to prioritize the main work.
                is_edition = 'edition' in desc or any('edition' in inst for inst in entity_data['instances'])
                edition_penalty = -1000 if is_edition else 0

                combined_score = title_score + author_score + instance_score + desc_score + edition_penalty
                
                candidate_data = {
                    'id': qid, 'description': result.get('description'), 'title_score': title_score,
                    'author_score': author_score, 'combined_score': combined_score,
                    'date': entity_data['date']
                }
                current_candidates.append(candidate_data)
                all_candidates.append(candidate_data) # Also add to the global list for final fallback

            # If we found any candidates in this batch, check for a winner
            if current_candidates:
                best_in_batch = max(current_candidates, key=lambda x: x['combined_score'])
                # Confidence check: be slightly more lenient on title if author match is strong.
                is_strong_author_match = best_in_batch['author_score'] > 95
                is_good_enough_title = best_in_batch['title_score'] > 75 if is_strong_author_match else best_in_batch['title_score'] > 80

                if is_good_enough_title and best_in_batch['author_score'] > 80:
                    # Final check to ensure we didn't pick a penalized edition
                    if best_in_batch['combined_score'] > 0:
                        logging.info(f"High-confidence match found early with query '{query}'. Halting search.")
                        return best_in_batch['id'], "Success", best_in_batch['description'], best_in_batch['date']

        except (ValueError, KeyError) as e:
            logging.error(f"Error processing search results for query '{query}': {e}")

    # --- Step 3: Final Evaluation if No Early Exit ---
    if not all_candidates:
        return None, "No Match Found", None, None

    # Find the best candidate from all combined search permutations
    best_candidate = max(all_candidates, key=lambda x: x['combined_score'])

    # Final confidence check, same as the one in the loop
    is_strong_author_match = best_candidate['author_score'] > 95
    is_good_enough_title = best_candidate['title_score'] > 75 if is_strong_author_match else best_candidate['title_score'] > 80

    if is_good_enough_title and best_candidate['author_score'] > 80 and best_candidate['combined_score'] > 0:
        logging.info(f"High-confidence match for '{book_title}': QID {best_candidate['id']} (T: {best_candidate['title_score']}, A: {best_candidate['author_score']})")
        return best_candidate['id'], "Success", best_candidate['description'], best_candidate['date']
    elif best_candidate['title_score'] > 85:
        logging.warning(f"Potential Author Mismatch for '{book_title}': QID {best_candidate['id']} (T: {best_candidate['title_score']}, A: {best_candidate['author_score']})")
        return None, "Author Mismatch", None, None
    else:
        logging.warning(f"Low confidence match for '{book_title}': QID {best_candidate['id']} (T: {best_candidate['title_score']}, A: {best_candidate['author_score']})")
        return None, "Low Confidence Match", None, None


def get_book_publication_date_robust(book_title: str, author_name: str) -> Tuple[str, str]:
    """
    [REFACTORED V3] Robustly fetches publication date.
    - Uses the highly efficient find_best_book_match.
    - Gets date and status in a single logical step.
    """
    cleaned_title = re.sub(r'\(.*?\)', '', book_title).strip()
    if not cleaned_title: return "Unknown", "Invalid Title"

    qid, status, description, date = find_best_book_match(cleaned_title, author_name)

    if qid and date:
        return date, "Success"
    elif qid:  # Match found but no structured date from the API
        # Fallback: check the description text for a year
        if description:
            year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', description)
            if year_match:
                logging.info(f"Found date '{year_match.group(1)}' in description for QID {qid}")
                return year_match.group(1), "Success (from description)"
        return "Unknown", "Date Not Found on Wikidata"

    return "Unknown", status

def load_processed_books(filename: str) -> Set[Tuple[str, str]]:
    """
    Loads all (title, author) pairs from the output file to prevent re-processing.
    This is key for making the script resumable.
    """
    if not os.path.exists(filename): return set()
    try:
        # We only need TITLE and AUTHOR to identify a processed book.
        df_processed = pd.read_csv(filename, usecols=['TITLE', 'AUTHOR'],
                                   # Specify dtype to avoid mixed type warnings
                                   dtype={'TITLE': str, 'AUTHOR': str})
        # Create a set of tuples for efficient 'in' checking.
        return set(zip(df_processed['TITLE'].astype(str), df_processed['AUTHOR'].astype(str)))
    except (Exception) as e:
        logging.warning(f"Could not load processed books from '{filename}'. Will re-process all. Error: {e}")
        return set()

def save_cache_to_file(cache: Dict, df_original: pd.DataFrame, output_file: str, is_production: bool):
    """Saves the contents of the book date cache to the output CSV."""
    if not cache:
        logging.info("Cache is empty. Nothing to save.")
        return

    logging.info(f"--- Saving batch of {len(cache)} unique books ---")
    try:
        results_df = pd.DataFrame.from_dict(cache, orient='index').reset_index()
        results_df.columns = ['TITLE', 'AUTHOR', 'publication_date', 'date_fetch_status']
        
        # Ensure correct dtypes to prevent merge errors
        df_original['TITLE'] = df_original['TITLE'].astype(str)
        df_original['AUTHOR'] = df_original['AUTHOR'].astype(str)
        results_df['TITLE'] = results_df['TITLE'].astype(str)
        results_df['AUTHOR'] = results_df['AUTHOR'].astype(str)

        final_df_batch = pd.merge(df_original, results_df, on=['TITLE', 'AUTHOR'], how='inner')

        if not final_df_batch.empty:
            file_exists = os.path.exists(output_file)
            output_mode = 'a' if is_production and file_exists else 'w'
            write_header = not (is_production and file_exists)
            final_df_batch.to_csv(output_file, mode=output_mode, header=write_header, index=False)
            logging.info(f"Successfully saved batch to '{output_file}'.")
        else:
            logging.warning("Batch was empty after merging. Nothing written to disk.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to save batch to '{output_file}'. Error: {e}")

# --- Main Execution ---
def main(input_file: str, output_file: str, sample_size: Optional[int], batch_size: int):
    """
    [REFACTORED V2] Main processing logic.
    - Unified workflow for both sample and full runs.
    - More efficient: processes unique books first, then maps results.
    - Saves progress in batches for safe, resumable execution.
    """
    start_time = time.time()
    
    try:
        df = pd.read_csv(input_file).dropna(subset=['TITLE', 'AUTHOR'])
        logging.info(f"Loaded {df.shape[0]} quotes from '{input_file}'.")
    except FileNotFoundError:
        logging.error(f"Input file not found at: {input_file}")
        return
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        return

    # --- Identify books to process ---
    
    # For resumability, skip books already in the output file
    processed_books = load_processed_books(output_file)
    if processed_books:
        logging.info(f"Loaded {len(processed_books)} unique books that are already in the output file. Skipping them.")
        # Create a temporary key for filtering
        df['temp_id'] = list(zip(df['TITLE'].astype(str), df['AUTHOR'].astype(str)))
        df = df[~df['temp_id'].isin(processed_books)].drop(columns=['temp_id'])

    if df.empty:
        logging.info("No new books to process. Exiting.")
        return

    # Identify unique books that need their dates fetched.
    unique_books_df = df[['TITLE', 'AUTHOR']].drop_duplicates().reset_index(drop=True)

    # If it's a sample run, take a slice of the unique books
    if sample_size is not None:
        if sample_size >= len(unique_books_df):
             logging.info(f"Sample size ({sample_size}) is >= unique books ({len(unique_books_df)}). Processing all.")
        else:
            logging.info(f"--- RUNNING IN SAMPLE MODE (processing {sample_size} unique books) ---")
            unique_books_df = unique_books_df.head(sample_size)
    else:
        logging.info("--- RUNNING IN PRODUCTION MODE (resumable) ---")
    
    total_books = len(unique_books_df)
    logging.info(f"Found {total_books} unique new books to process.")

    # --- Fetch data for unique books and save in batches ---
    book_date_cache = {}
    is_production = sample_size is None
    processed_count = 0

    try:
        for index, row in tqdm(unique_books_df.iterrows(), total=unique_books_df.shape[0], desc="Fetching dates for unique books"):
            book_title, author_name = row['TITLE'], row['AUTHOR']
            cache_key = (book_title, author_name)
            
            try:
                date, reason = get_book_publication_date_robust(book_title, author_name)
                book_date_cache[cache_key] = {'publication_date': date, 'date_fetch_status': reason}
                processed_count += 1
                
                # Calculate and log progress
                elapsed_time = time.time() - start_time
                if processed_count > 0:
                    avg_time_per_book = elapsed_time / processed_count
                    remaining_books = total_books - processed_count
                    estimated_remaining_time = remaining_books * avg_time_per_book
                    
                    logging.info(f"Processed {processed_count}/{total_books} ({processed_count/total_books*100:.1f}%) - "
                               f"'{book_title}' -> {date} ({reason}) - "
                               f"ETA: {estimated_remaining_time/3600:.1f}h")
                
                # Use adaptive delay based on current performance
                current_delay = max(MIN_DELAY_S, API_DELAY_S * (1 + consecutive_failures * 0.1))
                time.sleep(current_delay)

            except Exception as e:
                logging.error(f"Error processing book '{book_title}' by '{author_name}': {e}")
                continue

            # --- Save results in batches ---
            is_last_item = (index + 1) == len(unique_books_df)
            if (len(book_date_cache) >= batch_size) or (is_last_item and book_date_cache):
                save_cache_to_file(book_date_cache, df, output_file, is_production)
                # Clear the cache for the next batch
                book_date_cache = {}
                
    finally:
        # This block will execute on normal completion or if interrupted (e.g., Ctrl+C)
        logging.info("Script is finishing. Attempting to save any remaining data in cache...")
        save_cache_to_file(book_date_cache, df, output_file, is_production)

    total_time = time.time() - start_time
    logging.info(f"\nProcessing complete! Total time: {total_time/3600:.1f}h, "
                f"Processed {processed_count}/{total_books} books, "
                f"Average: {total_time/processed_count:.1f}s per book")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch book publication dates from Wikidata for a list of quotes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        default='data/interim/cleaning_data/Data/quotes_with_author_language.csv',
        help='Input CSV file containing quotes with TITLE and AUTHOR columns.'
    )
    parser.add_argument(
        '--output',
        default='data/interim/cleaning_data/Data/quotes_with_publication_dates.csv',
        help='Output CSV file for quotes with appended publication dates.'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Run in sample mode by processing N unique books. If not set, runs in production mode on all new books.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Number of unique books to process before saving results to disk. Smaller batches save progress more often.'
    )
    
    args = parser.parse_args()

    # Determine the correct output file based on whether it's a sample run
    output_filename = args.output
    if args.sample is not None:
        base, ext = os.path.splitext(output_filename)
        output_filename = f"{base}_SAMPLE_{args.sample}{ext}"

    main(
        input_file=args.input,
        output_file=output_filename,
        sample_size=args.sample,
        batch_size=args.batch_size
    ) 