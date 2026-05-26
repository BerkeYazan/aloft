import pandas as pd
import requests
import time
from thefuzz import fuzz
from tqdm import tqdm

# --- Configuration ---
WIKIDATA_API_URL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "MyThesisProject/1.0 (b.yazan@uu.nl)"
HEADERS = {'Accept': 'application/sparql-results+json', 'User-Agent': USER_AGENT}
SIMILARITY_THRESHOLD = 90
PERSON_KEYWORDS = [
    'author', 'writer', 'poet', 'novelist', 'playwright', 'essayist', 'journalist',
    'comedian', 'actor', 'screenwriter', 'philosopher', 'scientist', 'politician',
    'singer-songwriter', 'memoirist', 'diarist', 'lyricist', 'statesman'
]

def query_wikidata_by_id(author_id):
    query = f"""
    SELECT ?languageLabel ?countryLabel WHERE {{
      OPTIONAL {{ wd:{author_id} wdt:P103 ?language. }}
      OPTIONAL {{ wd:{author_id} wdt:P27 ?country. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 1"""
    try:
        response = requests.get(WIKIDATA_API_URL, params={'query': query}, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()['results']['bindings']
        if not data: return None, None
        if 'languageLabel' in data[0]: return data[0]['languageLabel']['value'], 'Native Language'
        if 'countryLabel' in data[0]: return data[0]['countryLabel']['value'], 'Country of Citizenship'
        return None, None
    except (requests.exceptions.RequestException, IndexError, KeyError):
        return None, None

def find_best_match(author_name):
    params = {'action': 'wbsearchentities', 'format': 'json', 'language': 'en', 'type': 'item', 'search': author_name}
    try:
        response = requests.get(WIKIDATA_SEARCH_URL, params=params, headers={'User-Agent': USER_AGENT}, timeout=10)
        response.raise_for_status()
        search_results = response.json().get('search', [])
        if not search_results: return None

        for result in search_results:
            label = result.get('label', '')
            score = fuzz.token_sort_ratio(author_name, label)
            description = result.get('description', '').lower()
            if any(keyword in description for keyword in PERSON_KEYWORDS) and score > SIMILARITY_THRESHOLD:
                return result.get('id')

        first_result = search_results[0]
        first_label = first_result.get('label', '')
        if fuzz.token_sort_ratio(author_name, first_label) > SIMILARITY_THRESHOLD:
            return first_result.get('id')
        return None
    except (requests.exceptions.RequestException, KeyError):
        return None

def get_author_data_robust(author_name):
    best_match_id = find_best_match(author_name)
    if best_match_id:
        data, method = query_wikidata_by_id(best_match_id)
        if data:
            return data, method
    return "Not Found", "N/A"

# --- Main Execution ---
if __name__ == "__main__":
    input_file = 'data/interim/cleaning_data/Data/quotes_cleaned_authors.csv'
    output_file = 'data/interim/cleaning_data/Data/quotes_with_author_language.csv'
    
    try:
        df = pd.read_csv(input_file)
        unique_authors = df['AUTHOR'].unique()
        
        print(f"Starting to fetch data for {len(unique_authors)} unique authors. This will take approximately 1.5-2 hours.")
        
        # Use a dictionary for caching results to speed up the process
        author_data_cache = {}
        
        for author in tqdm(unique_authors, desc="Fetching Author Data"):
            if author not in author_data_cache:
                language, method = get_author_data_robust(author)
                author_data_cache[author] = {'language_or_country': language, 'source_method': method}
                time.sleep(1) # Be polite to the API
        
        # Map the fetched data back to the original dataframe
        df['language_or_country'] = df['AUTHOR'].map(lambda x: author_data_cache[x]['language_or_country'])
        df['source_method'] = df['AUTHOR'].map(lambda x: author_data_cache[x]['source_method'])
        
        # Save the final, enriched dataframe
        df.to_csv(output_file, index=False)
        
        print("\nProcessing complete!")
        print(f"Enriched data saved to: {output_file}")

    except FileNotFoundError:
        print(f"Cleaned file not found at: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}") 