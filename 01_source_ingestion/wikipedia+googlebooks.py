import pandas as pd
import time
import requests
from fuzzywuzzy import fuzz
from tqdm import tqdm
from tabulate import tabulate
import json
import logging
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
WIKIDATA_SEARCH_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "MyThesisProject/1.0"
MAX_RETRIES = 5
INITIAL_BACKOFF = 1 # seconds
LOG_LEVEL = logging.INFO
PROCESSED_DATA_FILE = 'data/interim/cleaning_data/Data/processed_books_metadata.csv'
INPUT_FILE = 'data/interim/cleaning_data/Data/tags-goodreads-english-popular-quotes.csv'
BATCH_SIZE = 10

# --- Setup Logging ---
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    handlers=[logging.FileHandler("processing_activity.log"),
                              logging.StreamHandler()])


def make_api_request_with_retry(url, params=None, headers=None, method="GET", data=None, timeout=30):
    """Makes an HTTP request with retries and exponential backoff."""
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.request(method, url, params=params, headers=headers, data=data, timeout=timeout)
            if response.status_code == 429: # Too Many Requests
                retry_after = response.headers.get("Retry-After")
                sleep_time = INITIAL_BACKOFF * (2 ** attempt)
                if retry_after and retry_after.isdigit():
                    sleep_time = max(sleep_time, int(retry_after))
                logging.warning(f"Rate limit hit (429) for {url}. Retrying in {sleep_time}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(sleep_time)
                continue
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            return response
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout occurred for {url}. (Attempt {attempt + 1}/{MAX_RETRIES})")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed for {url}: {e}. (Attempt {attempt + 1}/{MAX_RETRIES})")
        
        if attempt < MAX_RETRIES - 1:
            sleep_time = INITIAL_BACKOFF * (2 ** attempt)
            logging.info(f"Waiting {sleep_time}s before next retry...")
            time.sleep(sleep_time)
        else:
            logging.error(f"All {MAX_RETRIES} retries failed for {url}.")
            return None # Or raise an exception

def search_wikidata_qid(title, author):
    """Searches Wikidata for a QID based on title, refines with author, and looks for YYYY edition in descriptions."""
    logging.debug(f"Searching QID for Title: '{title}', Author: '{author}'")
    headers = {"User-Agent": USER_AGENT}
    author_lower = author.lower() if pd.notna(author) and author.strip() else None

    search_titles = [title]
    if ":" in title: # Try a shorter title if a colon is present
        parts = title.split(":", 1) # Split only on the first colon
        part1 = parts[0].strip()
        part2 = parts[1].strip() if len(parts) > 1 else ""

        if part1.lower() != title.lower() and part1: # Avoid duplicate searches and empty strings
            search_titles.append(part1)
            logging.debug(f"DEBUG: Adding first part of title to search: '{part1}'")
        
        if part2.lower() != title.lower() and part2 and part2.lower() != part1.lower(): # Avoid duplicate, empty, or same-as-part1
            search_titles.append(part2)
            logging.debug(f"DEBUG: Adding second part of title to search: '{part2}'")

    all_search_results = []
    for s_title in search_titles:
        params_title_only = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item",
            "search": s_title
        }
        try:
            response = make_api_request_with_retry(WIKIDATA_SEARCH_API_URL, params=params_title_only)
            if not response:
                logging.warning(f"Failed to get response for title search '{s_title}' after retries.")
                continue

            logging.debug(f"Search API URL for '{s_title}': {response.url}")
            logging.debug(f"Search API Status Code for '{s_title}': {response.status_code}")
            response.raise_for_status()
            search_json = response.json()
            current_results = search_json.get("search", [])
            logging.debug(f"DEBUG: Found {len(current_results)} results for title query: '{s_title}'.")
            all_search_results.extend(current_results)
            if not current_results and response.status_code == 200:
                logging.debug(f"Title search API ('{s_title}') returned 0 results. Response: {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during Wikidata search for '{s_title}': {e}")
            time.sleep(0.3) # Add sleep even on error before trying next title
            continue # Try next search title if one exists
        except json.JSONDecodeError as e:
            response_text_snippet = response.text[:500] if response and hasattr(response, 'text') else 'No response text available'
            logging.error(f"Error decoding JSON for title search '{s_title}': {e}. Response text: {response_text_snippet}")
            time.sleep(0.3) # Add sleep even on error before trying next title
            continue
        time.sleep(0.3) # Respectful delay between multiple search queries for the same item

    if not all_search_results:
        logging.debug(f"No search results found for any title variations of '{title}'. Returning None.")
        return None, None, None

    # Deduplicate results based on QID, keeping the first encountered (usually from full title search first)
    seen_qids = set()
    unique_search_results = []
    for res in all_search_results:
        if res.get('id') not in seen_qids:
            unique_search_results.append(res)
            seen_qids.add(res.get('id'))
    logging.debug(f"DEBUG: Total unique search results after deduplication: {len(unique_search_results)}")

    priority1_matches = []  # 'by author' in desc + book type
    priority2_matches = []  # author in label + book type
    priority3_matches = []  # author in desc (general) + book type
    priority4_matches = []  # book type only
    edition_candidates = [] # For "YYYY edition" matches: (year, score, qid, label, description)

    book_type_terms = ["book", "novel", "literary work", "publication", "non-fiction book", "science book", "play", "poem", "poetry", 
                       "comic", "graphic novel", "novella", "short story", "short fiction", "comedy", "tragedy", "pastoral comedy"]
    for i, result in enumerate(unique_search_results): 
        if i < 5 : # Print details for a few results for logging
            logging.debug(f"Processing unique result {i} - ID: {result.get('id')}, Label: '{result.get('label')}', Desc: '{result.get('description')}'")
        
        qid = result.get("id")
        description_lower = result.get("description", "").lower()
        label_lower = result.get("label", "").lower()
        original_label = result.get("label", "") # Keep original case for matching
        
        is_book_type = any(term in description_lower for term in book_type_terms) or \
                       any(term in label_lower for term in book_type_terms) # Check label too for book type

        match_data = {"qid": qid, "label": original_label, "description": result.get("description", "")}

        # Try to parse "YYYY edition" from description if it's a book type
        if is_book_type:
            # Regex to find "YYYY edition", "YYYY [word] edition", or just a 4-digit year if it's a common pattern
            # This will match years between 1000 and 2999
            year_match = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b(?:\s+\w*)?\s+(?:edition|version|publication|issue|release)', description_lower)
            if not year_match: # Simpler regex if the above fails, just for a year if it looks like an edition description
                year_match = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', description_lower)
            
            if year_match:
                parsed_year = int(year_match.group(1))
                title_similarity_score = fuzz.ratio(title.lower(), original_label.lower())
                edition_candidates.append({
                    "year": parsed_year, 
                    "score": title_similarity_score, 
                    "qid": qid, 
                    "label": original_label,
                    "description": result.get("description", "")
                })
                logging.debug(f"QID {qid} ('{original_label}'): Found potential edition year {parsed_year} with score {title_similarity_score}.")

        if author_lower:
            by_author_in_desc = f"by {author_lower}" in description_lower or \
                                (description_lower.startswith(author_lower + "'s") and " by " not in description_lower) or \
                                (f"{author_lower}" == description_lower.split(" by ")[-1].strip() if " by " in description_lower else False)
            
            author_in_label = author_lower in label_lower
            author_in_desc_general = author_lower in description_lower

            if is_book_type:
                if by_author_in_desc:
                    logging.debug(f"QID {qid} ('{result.get('label')}'): Matched Priority 1 (by author: '{author_lower}' in desc + book type)")
                    priority1_matches.append(match_data)
                    continue 
                if author_in_label:
                    logging.debug(f"QID {qid} ('{result.get('label')}'): Matched Priority 2 (author: '{author_lower}' in label + book type)")
                    priority2_matches.append(match_data)
                    continue
                if author_in_desc_general:
                    logging.debug(f"QID {qid} ('{result.get('label')}'): Matched Priority 3 (author: '{author_lower}' in desc general + book type)")
                    priority3_matches.append(match_data)
                    continue
        
        if is_book_type:
            logging.debug(f"QID {qid} ('{result.get('label')}'): Matched Priority 4 (book type in desc/label)")
            priority4_matches.append(match_data)

    # Select the best QID based on priority and then fuzzy match on title
    best_primary_qid = None
    highest_primary_score = -1

    for p_level_name, p_level_matches in [("P1", priority1_matches), ("P2", priority2_matches), ("P3", priority3_matches), ("P4", priority4_matches)]:
        if p_level_matches:
            logging.debug(f"Found {len(p_level_matches)} matches in {p_level_name}")
            for match in p_level_matches:
                score = fuzz.ratio(title.lower(), match["label"].lower())
                logging.debug(f"{p_level_name} match QID {match['qid']} ('{match['label']}') - Title similarity score: {score}")
                if score > highest_primary_score:
                    highest_primary_score = score
                    best_primary_qid = match["qid"]
            if best_primary_qid: # If any match in this priority level, pick the best from it and stop
                logging.debug(f"Selected best primary QID from {p_level_name}: {best_primary_qid} with title score {highest_primary_score}")
                # We don't return immediately; we want to collect edition candidates too.
                break # Break from priority level loop, but continue to process editions
    
    best_edition_qid = None
    best_edition_year = None

    if edition_candidates:
        # Sort by year (ascending), then by score (descending)
        edition_candidates.sort(key=lambda x: (x["year"], -x["score"])) 
        # Filter for candidates with a reasonably high title match score (e.g. > 70) to avoid irrelevant editions
        # This threshold might need adjustment based on typical title variations
        MIN_EDITION_TITLE_SCORE = 70 
        plausible_edition_candidates = [cand for cand in edition_candidates if cand["score"] >= MIN_EDITION_TITLE_SCORE]
        
        if plausible_edition_candidates:
            best_edition_candidate = plausible_edition_candidates[0]
            best_edition_qid = best_edition_candidate["qid"]
            best_edition_year = best_edition_candidate["year"]
            logging.debug(f"Selected best edition candidate: QID {best_edition_qid} ('{best_edition_candidate['label']}'), Year {best_edition_year}, Score {best_edition_candidate['score']}")
        else:
            logging.debug("No edition candidates met the minimum title similarity score.")

    if not best_primary_qid and not best_edition_qid:
        logging.debug(f"No primary QID or suitable edition QID found for title '{title}'.")
        return None, None, None
    
    logging.debug(f"For title '{title}': Primary QID: {best_primary_qid} (Score: {highest_primary_score}), Edition QID: {best_edition_qid}, Edition Year: {best_edition_year}")
    return best_primary_qid, best_edition_qid, best_edition_year

def fetch_google_books_data(title, author, api_key=None):
    """Fetches publication year and categories (genres) from Google Books API."""
    logging.info(f"Attempting to fetch data for '{title}' by '{author}' from Google Books API.")
    base_url = "https://www.googleapis.com/books/v1/volumes"
    query_parts = []
    if title:
        query_parts.append(f"intitle:{title}")
    if author:
        query_parts.append(f"inauthor:{author}")
    
    if not query_parts:
        logging.warning("No title or author provided for Google Books search.")
        return None, None

    query = "+".join(query_parts)
    params = {"q": query, "maxResults": 1, "fields": "totalItems,items(volumeInfo/publishedDate,volumeInfo/categories)"} # Request specific fields
    if api_key:
        params["key"] = api_key
    else:
        logging.warning("GOOGLE_BOOKS_API_KEY is not set. API calls may be limited or fail.")

    pub_year_final = None
    genres_final = None

    try:
        response = make_api_request_with_retry(base_url, params=params)
        if not response:
            logging.error(f"Failed to get response from Google Books API for '{title}' after retries.")
            return None, None
            
        logging.debug(f"Google Books API URL: {response.url}")
        logging.debug(f"Google Books API Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if data.get("totalItems", 0) > 0 and "items" in data:
            item = data["items"][0]
            volume_info = item.get("volumeInfo", {})
            
            # Extract Publication Year
            published_date_raw = volume_info.get("publishedDate")
            if published_date_raw:
                year_str = str(published_date_raw)[:4]
                if year_str.isdigit() and len(year_str) == 4:
                    pub_year_final = year_str
                    logging.debug(f"Found publishedYear: '{pub_year_final}' for '{title}'")
                else:
                    logging.warning(f"Could not reliably extract year from '{published_date_raw}' for '{title}'.")
            else:
                logging.debug(f"'publishedDate' field not found in Google Books API response for '{title}'.")

            # Extract Categories (Genres)
            categories_list = volume_info.get("categories")
            if categories_list and isinstance(categories_list, list):
                genres_final = ", ".join(categories_list)
                logging.debug(f"Found categories: '{genres_final}' for '{title}'")
            else:
                logging.debug(f"'categories' field not found or not a list in Google Books API response for '{title}'.")
        else:
            logging.info(f"No items found in Google Books API response for '{title}'.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Google Books API for '{title}': {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from Google Books API for '{title}': {e}. Response: {response.text[:200] if response else 'No response'}")
    
    return pub_year_final, genres_final

def fetch_wikidata_details(qid):
    """Fetches publication date, original language, and genres for a Wikidata QID using SPARQL."""
    logging.debug(f"Fetching details for QID: {qid}")
    # SPARQL query refined to use COALESCE for language and publication date, and ensure robust fetching.
    query = f"""
    SELECT
      ?work
      (COALESCE(?pub_date_p577, ?pub_date_p580) AS ?publication_date)
      (COALESCE(?original_work_lang_label, ?direct_item_lang_label) AS ?orig_lang_label)
      (GROUP_CONCAT(DISTINCT ?genre_label; separator=", ") AS ?genres_labels)
    WHERE {{
      BIND(wd:{qid} AS ?item)
      BIND(?item AS ?work) # Bind ?work early for grouping

      # Publication Date - P577 (publication date) or P580 (start date)
      OPTIONAL {{ ?item wdt:P577 ?pub_date_p577. }}
      OPTIONAL {{ ?item wdt:P580 ?pub_date_p580. }}

      # Language from original work (if item is an edition/translation)
      OPTIONAL {{
        ?item wdt:P629 ?original_w. # P629: edition or translation of
        ?original_w wdt:P407 ?orig_lang_uri. # P407: language of work or name
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?orig_lang_uri rdfs:label ?original_work_lang_label. }}
      }}

      # Language directly from the item (used if not found via P629 or if item is not an edition)
      OPTIONAL {{
        # This block will be used if ?original_work_lang_label is unbound
        ?item wdt:P407 ?direct_lang_uri.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?direct_lang_uri rdfs:label ?direct_item_lang_label. }}
      }}

      # Genres from the item itself
      OPTIONAL {{
        ?item wdt:P136 ?genre_uri. # P136: genre
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?genre_uri rdfs:label ?genre_label. }}
      }}
    }}
    GROUP BY ?work ?pub_date_p577 ?pub_date_p580 ?original_work_lang_label ?direct_item_lang_label
    LIMIT 1
    """
    params = {"query": query, "format": "json"}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    try:
        response = make_api_request_with_retry(WIKIDATA_SPARQL_URL, params=params, headers=headers)
        if not response:
            logging.error(f"Failed to get response from SPARQL query for QID {qid} after retries.")
            return None, None, None

        logging.debug(f"SPARQL Query URL: {response.url}") 
        logging.debug(f"SPARQL Query Status Code: {response.status_code}")
        response.raise_for_status()
        sparql_json = response.json()
        results = sparql_json.get("results", {}).get("bindings", [])
        logging.debug(f"Found {len(results)} bindings from SPARQL query for QID {qid}.")
        if not results and response.status_code == 200:
            logging.debug(f"SPARQL query returned 0 results. Response text: {response.text[:500]}")

        if results:
            data = results[0]
            logging.debug(f"Raw SPARQL data for {qid}: {data}") # Full data for the row

            # Detailed debug for specific fields before .get('value')
            logging.debug(f"data.get('publication_date') for {qid}: {data.get('publication_date')}")
            logging.debug(f"data.get('orig_lang_label') for {qid}: {data.get('orig_lang_label')}")
            logging.debug(f"data.get('genres_labels') for {qid}: {data.get('genres_labels')}")

            pub_date = data.get("publication_date", {}).get("value")
            if pub_date and 'T' in pub_date:
                pub_date = pub_date.split('T')[0]
            
            # Extract only the year from pub_date
            if pub_date and len(str(pub_date)) >= 4:
                pub_year = str(pub_date)[:4]
                if pub_year.isdigit():
                    pub_date = pub_year
                    logging.debug(f"Wikidata PubYear successfully extracted: {pub_date}")
                else:
                    logging.warning(f"Wikidata pub_date '{pub_date}' did not start with a valid year. Setting to None.")
                    pub_date = None # Invalid year format
            elif pub_date: # It exists but is less than 4 chars, so invalid
                logging.warning(f"Wikidata pub_date '{pub_date}' is too short. Setting to None.")
                pub_date = None

            orig_lang = data.get("orig_lang_label", {}).get("value")
            genres = data.get("genres_labels", {}).get("value")

            # Normalize English language variations
            if orig_lang and orig_lang.lower() in ["british english", "american english"]:
                orig_lang = "English"
                logging.debug(f"Normalized language to 'English' for QID {qid}")

            logging.debug(f"Extracted for QID {qid} - PubDate: {pub_date}, Lang: {orig_lang}, Genres: {genres}")
            return pub_date, orig_lang, genres
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching details for QID {qid}: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON for QID {qid}: {e}. Response: {response.text[:500] if response else 'No response object'}")
    logging.debug(f"Returning None, None, None for QID {qid} details due to an issue.")
    return None, None, None

# --- Main Processing ---

def load_processed_ids(filename):
    """Loads already processed book identifiers (e.g., a composite of title and author) to avoid re-processing."""
    processed_ids = set()
    if os.path.exists(filename):
        try:
            df_processed = pd.read_csv(filename)
            # Create a unique identifier for each processed row to check against
            # This assumes 'TITLE' and 'AUTHOR' uniquely identify a book for processing purposes
            # Handle potential missing authors by filling with a placeholder for consistent ID generation
            df_processed['ProcessedID'] = df_processed['TITLE'].astype(str) + "||" + df_processed['AUTHOR'].fillna('N/A').astype(str)
            processed_ids.update(df_processed['ProcessedID'])
            logging.info(f"Loaded {len(processed_ids)} already processed book IDs from '{filename}'.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Processed data file '{filename}' is empty. Starting fresh.")
        except Exception as e:
            logging.error(f"Error loading processed IDs from '{filename}': {e}. Consider deleting or fixing the file.")
            # Potentially raise the error or exit if loading processed IDs is critical
    return processed_ids

def main():
    logging.info("Starting metadata fetching process.")

    try:
        full_df = pd.read_csv(INPUT_FILE)
        logging.info(f"Successfully loaded input data from '{INPUT_FILE}'. Total rows: {len(full_df)}")
    except FileNotFoundError:
        logging.error(f"Input file '{INPUT_FILE}' not found. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading input CSV '{INPUT_FILE}': {e}. Exiting.")
        return

    # For testing, you might want to work with a smaller sample initially
    full_df = full_df.sample(100, random_state=412).reset_index(drop=True) # Uncomment for testing
    # logging.info(f"Using a sample of {len(full_df)} rows for processing.")

    # Randomize the order of the input DataFrame
    full_df = full_df.sample(frac=1, random_state=412).reset_index(drop=True) # random_state for reproducibility
    logging.info(f"Randomized the order of {len(full_df)} input rows.")

    processed_ids = load_processed_ids(PROCESSED_DATA_FILE) # This will now always find no file or an empty one
    
    # Filter out already processed rows from full_df
    # Create the same 'ProcessedID' for the input dataframe
    full_df['ProcessedID_temp'] = full_df['TITLE'].astype(str) + "||" + full_df['AUTHOR'].fillna('N/A').astype(str)
    df_to_process = full_df[~full_df['ProcessedID_temp'].isin(processed_ids)].copy() # Use .copy() to avoid SettingWithCopyWarning
    df_to_process.drop(columns=['ProcessedID_temp'], inplace=True) # Drop the temporary column

    if df_to_process.empty:
        logging.info("No new books to process. All entries from input file are already in the processed data file.")
        # Optionally, print the head of the existing processed data if needed
        if os.path.exists(PROCESSED_DATA_FILE):
            try:
                existing_df = pd.read_csv(PROCESSED_DATA_FILE)
                logging.info(f"\n\n--- Head of the existing results table ('{PROCESSED_DATA_FILE}') ---")
                print(tabulate(existing_df.head(), headers='keys', tablefmt='psql')) # Using print for direct table output
            except Exception as e:
                logging.error(f"Could not read or display existing processed file: {e}")
        return

    logging.info(f"Starting processing for {len(df_to_process)} new books (out of {len(full_df)} total).")

    for start_index in range(0, len(df_to_process), BATCH_SIZE):
        end_index = start_index + BATCH_SIZE
        batch_df = df_to_process.iloc[start_index:end_index]
        logging.info(f"Processing batch: rows {start_index + 1} to {min(end_index, len(df_to_process))} (out of {len(df_to_process)} new rows).")
        
        batch_results_data = []

        for _, row in tqdm(batch_df.iterrows(), total=batch_df.shape[0], desc=f"Batch {start_index//BATCH_SIZE + 1}"):
            title_val = row["TITLE"]
            author_val = row["AUTHOR"]
            
            primary_qid, edition_qid, edition_year = search_wikidata_qid(title_val, author_val)
            time.sleep(0.5)  # Inter-book delay for Wikidata search

            pub_year_wd, original_language_wd, genres_wd = None, None, None
            final_qid_to_use = primary_qid # Default to primary QID

            if primary_qid:
                logging.debug(f"Attempting to fetch details for primary QID: {primary_qid} for '{title_val}'")
                pub_year_wd, original_language_wd, genres_wd = fetch_wikidata_details(primary_qid)
                time.sleep(0.5) # Inter-detail fetch delay

            # If primary QID didn't yield a year, and a suitable edition candidate exists
            if pub_year_wd is None and edition_qid and edition_year:
                logging.info(f"Primary QID {primary_qid} for '{title_val}' did not yield a publication year. Trying edition candidate QID {edition_qid} with year {edition_year}.")
                # Fetch details for the edition QID (might overwrite language/genres if primary QID gave them but not year)
                # Or, only fetch if language/genres are also None, or if edition_qid is different from primary_qid
                # For simplicity now, always fetch if we are using the edition_year.
                edition_lang, edition_genres = None, None # Initialize
                _temp_year, edition_lang, edition_genres = fetch_wikidata_details(edition_qid)
                time.sleep(0.5)

                pub_year_wd = edition_year # Use the parsed year from description
                original_language_wd = edition_lang if edition_lang else original_language_wd # Prefer edition lang if found
                genres_wd = edition_genres if edition_genres else genres_wd # Prefer edition genres if found
                final_qid_to_use = edition_qid
                logging.debug(f"Using edition data for '{title_val}': QID={final_qid_to_use}, Year={pub_year_wd}, Lang={original_language_wd}, Genres={genres_wd}")

            # Fallback to Google Books API
            pub_year_gb, genres_gb = None, None
            # Condition to try Google Books: No QID, OR QID found but missing pub_year from Wikidata
            if not final_qid_to_use or (final_qid_to_use and pub_year_wd is None):
                if not final_qid_to_use:
                    logging.info(f"No QID found for '{title_val}'. Trying Google Books API.")
                else: # final_qid_to_use exists but pub_year_wd is missing
                    logging.info(f"Wikidata QID {final_qid_to_use} found for '{title_val}', but missing publication year ('{pub_year_wd}'). Trying Google Books API for year and genres.")
                pub_year_gb, genres_gb = fetch_google_books_data(title_val, author_val, api_key=GOOGLE_BOOKS_API_KEY)
                if pub_year_gb or genres_gb: # Add a small delay if Google Books API was called
                    time.sleep(0.3) 

            # Consolidate results (Wikidata preferred, then Google Books)
            final_pub_year = pub_year_wd if pub_year_wd else pub_year_gb
            final_original_language = original_language_wd # Google Books doesn't provide this easily
            final_genres = genres_wd if genres_wd else genres_gb # Prioritize Wikidata genres

            if pub_year_wd is None and pub_year_gb:
                logging.debug(f"Using Google Books pub_year: {pub_year_gb} for '{title_val}'")
            if not genres_wd and genres_gb:
                logging.debug(f"Using Google Books genres: '{genres_gb}' for '{title_val}'")
            
            # Append all original columns from the row, plus the new ones
            new_row_data = row.to_dict()
            # new_row_data["wikidataQID"] = final_qid_to_use # Removed as per request
            new_row_data["publishedDate"] = final_pub_year
            new_row_data["originalLanguage"] = final_original_language
            new_row_data["genres"] = final_genres
            batch_results_data.append(new_row_data)

        # After processing the batch, convert to DataFrame and save
        if batch_results_data:
            results_df_batch = pd.DataFrame(batch_results_data)
            
            # Define desired column order for robustness
            if not results_df_batch.empty:
                # Get original columns from the first row, assuming all rows in batch have same original structure
                # Exclude the new columns we are about to add, in case they were somehow in original_cols by mistake
                first_row_keys = list(batch_results_data[0].keys())
                new_metadata_cols = ["publishedDate", "originalLanguage", "genres"]
                original_cols = [col for col in first_row_keys if col not in new_metadata_cols]
                
                desired_columns = original_cols + new_metadata_cols
                
                # Ensure all desired columns exist in the DataFrame, fill with NaN if any are missing (shouldn't happen with current logic but safe)
                for col in desired_columns:
                    if col not in results_df_batch.columns:
                        results_df_batch[col] = pd.NA 
                results_df_batch = results_df_batch.reindex(columns=desired_columns)
            
            CSV_WRITE_RETRIES = 3
            CSV_WRITE_BACKOFF = 2 # seconds

            for attempt in range(CSV_WRITE_RETRIES):
                try:
                    file_exists = os.path.exists(PROCESSED_DATA_FILE)
                    results_df_batch.to_csv(PROCESSED_DATA_FILE, 
                                            mode='a' if file_exists else 'w', 
                                            header=not file_exists, 
                                            index=False)
                    logging.info(f"Successfully appended batch {start_index//BATCH_SIZE + 1} results ({len(results_df_batch)} rows) to '{PROCESSED_DATA_FILE}'.")
                    break # Break from retry loop on success
                except Exception as e:
                    logging.error(f"Error writing batch results to '{PROCESSED_DATA_FILE}' (Attempt {attempt + 1}/{CSV_WRITE_RETRIES}): {e}")
                    if attempt < CSV_WRITE_RETRIES - 1:
                        logging.info(f"Retrying CSV write in {CSV_WRITE_BACKOFF}s...")
                        time.sleep(CSV_WRITE_BACKOFF)
                    else:
                        logging.error(f"All {CSV_WRITE_RETRIES} attempts to write batch to CSV failed. Data for this batch may be lost from the main file.")
                        # Current behavior: log and continue to the next batch.
                        # Alternative: save to a temporary backup file here if desired.

        # Optional: short pause between batches if APIs are very sensitive
        time.sleep(0.5) # Inter-batch delay for API sensitivity

    logging.info("Finished processing all batches.")

    # Final inspection of the full (or newly augmented) results file
    try:
        final_df = pd.read_csv(PROCESSED_DATA_FILE)
        pd.set_option('display.max_colwidth', None) 
        logging.info(f"\n\n--- Head of the final results table ('{PROCESSED_DATA_FILE}', {len(final_df)} rows total) ---")
        print(tabulate(final_df.head(), headers='keys', tablefmt='psql')) # Using print for direct table output
        logging.info(f"\nFull results saved to '{PROCESSED_DATA_FILE}'")
    except FileNotFoundError: # If no data was processed and file doesn't exist
        logging.info(f"No data processed or file '{PROCESSED_DATA_FILE}' not found. Nothing to display.")
    except Exception as e:
        logging.error(f"Could not read or display final processed file '{PROCESSED_DATA_FILE}': {e}")


if __name__ == "__main__":
    main()

