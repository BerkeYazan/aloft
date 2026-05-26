import pandas as pd
import os
import logging

def remove_incorrect_entries():
    """
    Removes specific, incorrectly processed entries from the extracted_text.csv
    so they can be re-processed correctly.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(script_dir, 'extracted_text.csv')

    # Setup basic logging for this script
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not os.path.exists(output_csv):
        logging.info("The 'extracted_text.csv' file does not exist. Nothing to remove.")
        return

    try:
        df = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
        logging.info("The 'extracted_text.csv' file is empty. Nothing to remove.")
        return
    except Exception as e:
        logging.error(f"Could not read the CSV file: {e}")
        return

    if 'Filename' not in df.columns:
        logging.error("The CSV file does not have a 'Filename' column. Cannot proceed.")
        return

    # List of exact prefixes for the files that were processed with wrong coordinates
    incorrect_prefixes = [
        'Screenshot 2025-07-12',
        'Screenshot 2025-07-15',
        'Screenshot 2025-07-16'
    ]

    initial_row_count = len(df)
    logging.info(f"Loaded {initial_row_count} rows from '{os.path.basename(output_csv)}'.")

    # Create a boolean mask to identify the rows to remove
    # Note: Using .startswith with a tuple of prefixes is more efficient
    mask_to_remove = df['Filename'].str.startswith(tuple(incorrect_prefixes))

    # Invert the mask to keep all rows that *don't* start with these prefixes
    df_filtered = df[~mask_to_remove]
    
    final_row_count = len(df_filtered)
    rows_removed = initial_row_count - final_row_count

    if rows_removed > 0:
        logging.info(f"Identified and marked {rows_removed} rows for removal.")
        try:
            # Overwrite the original file with the filtered data
            df_filtered.to_csv(output_csv, index=False)
            logging.info(f"Successfully removed {rows_removed} rows. The file has been updated.")
            logging.info("You can now run 'process_screenshots.py' again to re-process these files with the correct coordinates.")
        except Exception as e:
            logging.error(f"Failed to write the updated CSV file: {e}")
    else:
        logging.info("No rows matching the incorrect prefixes were found. No changes were made.")

if __name__ == '__main__':
    remove_incorrect_entries() 