import pandas as pd
import requests
import time
from thefuzz import fuzz
import re

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
    """
    Fetches author data from Wikidata using their QID.
    It first tries to get native language (P103). If that fails, it falls back
    to country of citizenship (P27).
    """
    # Query for both properties at once to be efficient
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
        if not data:
            return None, None
        
        # Prioritize native language
        if 'languageLabel' in data[0]:
            return data[0]['languageLabel']['value'], 'Native Language'
        # Fallback to country of citizenship
        if 'countryLabel' in data[0]:
            return data[0]['countryLabel']['value'], 'Country of Citizenship'
            
        return None, None
    except (requests.exceptions.RequestException, IndexError, KeyError):
        return None, None

def find_best_match(author_name):
    """
    Uses Wikidata's search API and fuzzy matching to find the best author match.
    First, it looks for a high-confidence match with a relevant description.
    If that fails, it falls back to trusting the first result if its name is similar enough.
    """
    params = {
        'action': 'wbsearchentities',
        'format': 'json', 'language': 'en', 'type': 'item', 'search': author_name
    }
    try:
        response = requests.get(WIKIDATA_SEARCH_URL, params=params, headers={'User-Agent': USER_AGENT}, timeout=10)
        response.raise_for_status()
        search_results = response.json().get('search', [])
        
        if not search_results:
            return None

        # --- High-Confidence Pass ---
        # First, try to find a perfect match with a descriptive keyword
        for result in search_results:
            label = result.get('label', '')
            score = fuzz.token_sort_ratio(author_name, label)
            
            description = result.get('description', '').lower()
            if any(keyword in description for keyword in PERSON_KEYWORDS) and score > SIMILARITY_THRESHOLD:
                # Found a great match, return it immediately
                return result.get('id')

        # --- Fallback Pass ---
        # If no high-confidence match was found, trust the first result if the name is similar
        first_result = search_results[0]
        first_label = first_result.get('label', '')
        first_score = fuzz.token_sort_ratio(author_name, first_label)
        if first_score > SIMILARITY_THRESHOLD:
            return first_result.get('id')

        return None # Could not find a suitable match
    except (requests.exceptions.RequestException, KeyError):
        return None

def get_author_data_robust(author_name):
    """Robustly finds author's language or country using a fallback mechanism."""
    best_match_id = find_best_match(author_name)
    
    if best_match_id:
        data, method = query_wikidata_by_id(best_match_id)
        if data:
            return data, method

    return "Not Found", "N/A"

def get_problematic_authors(authors_series):
    """
    Analyzes a Series of author names and categorizes them based on specific
    formatting issues.
    """
    problem_lists = {
        'Contains Parentheses': [],
        'No Spaces Between Letters': [],
        'Contains Digits': [],
        'Too Short (<4 chars)': []
    }

    no_space_regex = re.compile(r'^[a-zA-Z]{5,}[a-zA-Z]+$')

    for name in authors_series:
        # Rule 1: Contains parentheses
        if '(' in name or ')' in name:
            problem_lists['Contains Parentheses'].append(name)
        
        # Rule 2: No spaces in a multi-letter name
        if no_space_regex.match(name):
            problem_lists['No Spaces Between Letters'].append(name)
            
        # Rule 3: Contains digits
        if any(char.isdigit() for char in name):
            problem_lists['Contains Digits'].append(name)
            
        # Rule 4: Too short
        if len(name) < 4:
            problem_lists['Too Short (<4 chars)'].append(name)
            
    return problem_lists

def run_author_health_check(authors_series):
    """
    Analyzes a Series of author names and flags them based on a set of rules
    to identify potentially malformed or problematic names.
    """
    health_report = {
        'total_unique_authors': len(authors_series),
        'contains_parentheses': 0,
        'no_spaces_in_name': 0,
        'contains_digits': 0,
        'potentially_problematic': set()
    }
    no_space_regex = re.compile(r'^[a-zA-Z]{5,}[a-zA-Z]+$')
    for name in authors_series:
        is_problematic = False
        if '(' in name or ')' in name:
            health_report['contains_parentheses'] += 1
            is_problematic = True
        if no_space_regex.match(name):
            health_report['no_spaces_in_name'] += 1
            is_problematic = True
        if any(char.isdigit() for char in name):
            health_report['contains_digits'] += 1
            is_problematic = True
        if is_problematic:
            health_report['potentially_problematic'].add(name)
    return health_report

def get_parentheses_authors(authors_series):
    """Finds all author names in a Series that contain parentheses."""
    parentheses_authors = []
    for name in authors_series:
        if '(' in name or ')' in name:
            parentheses_authors.append(name)
    return sorted(parentheses_authors)

def clean_author_names(df):
    """
    Applies a series of cleaning steps to the author column and reports on changes.
    """
    report = {
        'authors_removed': [],
        'names_manually_fixed': {},
        'parentheses_cleaned': {}
    }

    # --- 1. Remove specific authors ---
    authors_to_remove = [
        "19", "2 Minute Insight", "50 Cent", "anonymous",
        "Transcendence", "Seinfeld 2000", "VIZ", "Avi", "ANONYMOUS"
    ]
    report['authors_removed'] = authors_to_remove
    df_cleaned = df[~df['AUTHOR'].isin(authors_to_remove)].copy()

    # --- 2. Perform manual-style cleaning on specific names ---
    manual_replacements = {
        "Alcott Louisa May 1832-1888": "Alcott Louisa May",
        "Frances Hodgson Burnett 1849-1924": "Frances Hodgson Burnett",
        "George R.R. Martin 2005": "George R.R. Martin",
        "Jason Fried David Heinemeier Hansson Matthew Linderman 37 Signals": "Jason Fried, David Heinemeier Hansson, Matthew Linderman",
    }
    report['names_manually_fixed'] = manual_replacements
    df_cleaned['AUTHOR'] = df_cleaned['AUTHOR'].replace(manual_replacements)

    # --- 3. Automatically clean parentheses ---
    # Create a copy of the original authors before cleaning for the report
    original_authors = df_cleaned['AUTHOR'].copy()
    # Regex to remove content in parentheses and trailing/leading spaces
    df_cleaned['AUTHOR'] = df_cleaned['AUTHOR'].str.replace(r'\(.*\)', '', regex=True).str.strip()
    
    # Find which names were changed to create the report
    changed_mask = original_authors != df_cleaned['AUTHOR']
    changed_authors = original_authors[changed_mask]
    for original_name in changed_authors:
        cleaned_name = df_cleaned[original_authors == original_name]['AUTHOR'].iloc[0]
        report['parentheses_cleaned'][original_name] = cleaned_name
        
    return df_cleaned, report

def generate_report_text(report):
    """Generates a string summary of the cleaning actions."""
    report_text = "--- Data Cleaning Report ---\n\n"
    
    report_text += f"1. Total Authors Removed: {len(report['authors_removed'])}\n"
    report_text += "-----------------------------------\n"
    for name in report['authors_removed']:
        report_text += f"- {name}\n"
        
    report_text += f"\n\n2. Specific Names Manually Fixed: {len(report['names_manually_fixed'])}\n"
    report_text += "-----------------------------------\n"
    for original, new in report['names_manually_fixed'].items():
        report_text += f"- '{original}' -> '{new}'\n"
        
    report_text += f"\n\n3. Names with Parentheses Cleaned: {len(report['parentheses_cleaned'])}\n"
    report_text += "-----------------------------------\n"
    for original, new in report['parentheses_cleaned'].items():
        report_text += f"- '{original}' -> '{new}'\n"
        
    return report_text

# --- Main Execution ---
if __name__ == "__main__":
    input_file = 'data/interim/cleaning_data/Data/tags-goodreads-english-popular-quotes.csv'
    output_csv = 'data/interim/cleaning_data/Data/quotes_cleaned_authors.csv'
    output_report = 'data/interim/misc/cleaning_report.txt'

    try:
        df = pd.read_csv(input_file)
        
        cleaned_df, report_data = clean_author_names(df)
        report_text = generate_report_text(report_data)
        
        # Save the cleaned data and the report
        cleaned_df.to_csv(output_csv, index=False)
        with open(output_report, 'w') as f:
            f.write(report_text)
            
        print(f"Cleaning complete.")
        print(f"Cleaned data saved to: {output_csv}")
        print(f"Cleaning report saved to: {output_report}")
        print("\n--- Summary from Report ---")
        print(f"- Authors Removed: {len(report_data['authors_removed'])}")
        print(f"- Names Fixed: {len(report_data['names_manually_fixed'])}")
        print(f"- Parentheses Cleaned: {len(report_data['parentheses_cleaned'])}")

    except FileNotFoundError:
        print(f"File not found at: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
