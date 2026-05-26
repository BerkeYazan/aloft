import pandas as pd
import webbrowser
import logging
import sys
import threading
from pynput import keyboard
import os
import re
import subprocess
import time
import random

# Platform-specific imports for waiting for a keypress
try:
    import tty, termios
except ImportError:
    # This will fail on Windows, so a fallback would be needed for cross-platform.
    # It's suitable for the user's macOS environment.
    pass

# --- Configuration ---
INPUT_FILE = 'data/interim/google_books_work/google_books_sample_output.csv'
SCREENSHOTS_FOLDER = 'data/interim/google_books_work/Snippet_Screenshots'
PROGRESS_FILE = 'data/interim/google_books_work/screenshot_progress.txt'

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# --- Global variables for keyboard listener ---
continue_event = threading.Event()
should_quit = False
should_refresh = False

def on_press(key):
    """Callback function for the keyboard listener."""
    global should_quit, should_refresh
    try:
        # Check for '<' to continue
        if key.char == '<':
            print("\nProcessing... Please wait.")
            continue_event.set()
            return False  # Stop the listener

        # Check for '1' to refresh page
        if key.char == '1':
            print("\nRefreshing page with a new random number...")
            should_refresh = True
            continue_event.set()
            return False

        # Check for '-' to quit
        if key.char == '-':
            print("\nQuit command received.")
            should_quit = True
            continue_event.set()
            return False  # Stop the listener
    except AttributeError:
        pass # Ignore special keys that don't have a 'char' attribute

def wait_for_user_action():
    """
    Waits for the user to press '<' to continue, or '-' to quit.
    Listens globally across the OS. This function does not print any prompts.
    """
    global should_quit, should_refresh
    should_quit = False
    should_refresh = False
    continue_event.clear()

    # Start listener in a non-blocking way
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Wait here until the event is set by the listener
    continue_event.wait()
    listener.join() # Clean up the listener thread

    if should_quit:
        return 'quit'
    if should_refresh:
        return 'refresh'
    return 'continue'


def sanitize_filename(name):
    """Sanitizes a string to be a valid filename."""
    name = str(name).strip().replace(' ', '_')
    # Remove characters that are not alphanumeric, underscore, hyphen, or period.
    name = re.sub(r'(?u)[^-\w.]', '', name)
    # Limit length to avoid issues with filesystems
    return name[:150]


def execute_applescript(script):
    """Executes an AppleScript command and returns its output."""
    if sys.platform != "darwin":
        logging.warning("AppleScript can only be run on macOS.")
        return None
    try:
        process = subprocess.Popen(
            ['osascript', '-e', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            # Don't log an error if it's just that the window/tab doesn't exist
            if "doesn’t understand the “close” message" in stderr:
                 pass
            else:
                logging.error(f"AppleScript error: {stderr.strip()}")
            return None
        return stdout.strip()
    except FileNotFoundError:
        logging.error("`osascript` command not found. Cannot run AppleScript.")
        return None
    except Exception as e:
        logging.error(f"An exception occurred while running AppleScript: {e}")
        return None


def open_preview_links():
    """
    Manages a two-tab system in Chrome to allow for efficient, sequential screenshotting of book previews.
    """
    logging.info(f"Starting screenshot collection process from '{INPUT_FILE}'.")

    # Ensure the output directory for screenshots exists
    try:
        os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create screenshot directory '{SCREENSHOTS_FOLDER}': {e}. Exiting.")
        return

    try:
        df = pd.read_csv(INPUT_FILE, sep=',')
    except FileNotFoundError:
        logging.error(f"Input file not found: '{INPUT_FILE}'. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading '{INPUT_FILE}': {e}. Exiting.")
        return

    required_cols = ['TITLE', 'AUTHOR', 'preview_link']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Input CSV must contain the columns: {', '.join(required_cols)}. Exiting.")
        return

    books_df = df[required_cols].dropna().reset_index(drop=True)
    total_books_in_file = len(books_df)

    if total_books_in_file < 2:
        logging.error("This script requires at least two books in the CSV to function. Exiting.")
        return

    # --- Read progress to allow resuming ---
    start_index = 0
    try:
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content:
                last_processed_index = int(content)
                start_index = last_processed_index + 1
    except (FileNotFoundError, ValueError, IndexError):
        pass # Ignore if file not found or invalid, start from 0

    if start_index >= total_books_in_file:
        logging.info("All books have already been processed.")
        return
    
    logging.info(f"Resuming from index {start_index} ({total_books_in_file - start_index} remaining).")
    
    # --- Explain the new process ---
    print("\n" + "="*80)
    print("--- IMPORTANT: NEW WORKFLOW ---")
    print("This script will create a NEW, dedicated INCOGNITO Google Chrome window with two tabs.")
    print("All work will be done in this window. Your focus will be switched automatically.")
    print("Please do not close this window or the tabs manually during the process.")
    print("Press Enter to continue...")
    input()

    # --- Setup the Chrome Window and Tabs ---
    logging.info("Setting up a new Chrome incognito window with two tabs...")

    # Step 1: Create the window and get its unique ID
    create_window_script = '''
    tell application "Google Chrome"
        set new_window to make new window with properties {mode: "incognito"}
        return id of new_window
    end tell
    '''
    window_id = execute_applescript(create_window_script)

    if not window_id:
        logging.error("Failed to create a new incognito window. Please ensure Chrome is running and permissions are set. Aborting.")
        return
    
    logging.info(f"Created new incognito window with ID: {window_id}")

    # Step 2: Use the window ID to set up the two tabs, reusing the initial tab.
    url1 = books_df.iloc[start_index]['preview_link']
    url2 = books_df.iloc[start_index + 1]['preview_link']
    
    # Set the URL of the first, pre-existing tab
    execute_applescript(f'tell application "Google Chrome" to set URL of tab 1 of window id {window_id} to "{url1}"')
    time.sleep(1) # Give it time to load

    # Create the second tab
    execute_applescript(f'tell application "Google Chrome" to tell window id {window_id} to make new tab with properties {{URL:"{url2}"}}')
    time.sleep(1)

    # Set focus to the first tab to begin
    execute_applescript(f'tell application "Google Chrome" to set active tab index of window id {window_id} to 1')
    active_tab_local_idx = 1

    # --- Main processing loop ---
    index = start_index
    while index < total_books_in_file:
        book = books_df.iloc[index]
        title, author = book['TITLE'], book['AUTHOR']
        
        print("\n" + "="*80)
        logging.info(f"NOW PROCESSING BOOK {index + 1}/{total_books_in_file}: '{title}' by {author}")
        
        tab_to_update_idx = active_tab_local_idx
        tab_to_focus_idx = 2 if active_tab_local_idx == 1 else 1
        
        try:
            before_files = set(os.listdir(SCREENSHOTS_FOLDER))
        except Exception as e:
            logging.error(f"Could not read screenshot directory: {e}. Skipping.")
            index += 1
            continue
        
        while True:
            print("The tab for the current book should be active in the new window.")
            print("1. Take your screenshot(s).")
            print("2. When finished, press < to switch to the next book.")
            print("3. If page is unavailable, press 1 to try a new random page.")
            sys.stdout.flush()

            action = wait_for_user_action()

            if action == 'quit':
                logging.info("Quit command received. Progress saved.")
                break

            if action == 'refresh':
                logging.info("Attempting to refresh with a new page number...")
                get_url_script = f'tell application "Google Chrome" to return URL of active tab of window id {window_id}'
                current_url = execute_applescript(get_url_script)

                if not current_url:
                    logging.warning("Could not get the current URL of the active tab.")
                    continue

                match = re.search(r'[?&]pg=PA(\d+)', current_url)
                if match:
                    current_page = int(match.group(1))
                    new_page = 0
                    if current_page < 10:
                        # If page is low, reroll up to 40, can be bigger.
                        new_page = random.randint(6, 40)
                        logging.info(f"Current page is {current_page} (<10). Rerolling to a random page in [6, 40]: {new_page}.")
                    else: # current_page >= 10
                        # Original logic: reroll to a smaller page.
                        new_page = random.randint(6, current_page - 1)
                        logging.info(f"Current page is {current_page} (>=10). Rerolling to a random page in [6, {current_page - 1}]: {new_page}.")

                    new_url = re.sub(r'pg=PA\d+', f'pg=PA{new_page}', current_url)
                    execute_applescript(f'tell application "Google Chrome" to set URL of active tab of window id {window_id} to "{new_url}"')
                else:
                    logging.warning("Could not find a page number (e.g., &pg=PA273) in the current URL.")
                continue

            if action == 'continue':
                break
        
        if action == 'quit':
            break

        # --- Process screenshots ---
        try:
            after_files = set(os.listdir(SCREENSHOTS_FOLDER))
            new_files = sorted(list(after_files - before_files))
        except Exception as e:
            logging.error(f"Could not read screenshot directory after action: {e}. Skipping this book.")
            index += 1
            continue

        if new_files:
            logging.info(f"Detected {len(new_files)} new screenshot(s). Renaming now...")
            def get_creation_time(filename):
                try: return os.stat(os.path.join(SCREENSHOTS_FOLDER, filename)).st_birthtime
                except: return float('inf')
            new_files.sort(key=get_creation_time)
            for filename in new_files:
                try:
                    old_path = os.path.join(SCREENSHOTS_FOLDER, filename)
                    time_str = time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime(os.stat(old_path).st_birthtime))
                    file_extension = os.path.splitext(filename)[1]
                    new_filename, counter = f"{time_str}{file_extension}", 1
                    new_path = os.path.join(SCREENSHOTS_FOLDER, new_filename)
                    while os.path.exists(new_path):
                        new_filename = f"{time_str}_{counter}{file_extension}"
                        new_path = os.path.join(SCREENSHOTS_FOLDER, new_filename)
                        counter += 1
                    os.rename(old_path, new_path)
                    logging.info(f"  -> Renamed '{filename}' to '{new_filename}'")
                except Exception as e:
                    logging.error(f"  -> Failed to rename '{filename}': {e}")
        else:
            logging.warning("No new screenshots were detected for this book.")
        
        # --- Save progress ---
        try:
            with open(PROGRESS_FILE, 'w') as f: f.write(str(index))
            logging.info(f"Progress saved. Last processed index: {index}")
        except IOError as e:
            logging.error(f"Failed to save progress: {e}")

        # --- Update URL of the background tab and switch focus ---
        next_book_load_idx = index + 2
        if next_book_load_idx < total_books_in_file:
            next_url = books_df.iloc[next_book_load_idx]['preview_link']
            logging.info(f"Loading '{books_df.iloc[next_book_load_idx]['TITLE']}' into the background tab.")
            execute_applescript(f'tell application "Google Chrome" to set URL of tab {tab_to_update_idx} of window id {window_id} to "{next_url}"')
        
        execute_applescript(f'tell application "Google Chrome" to set active tab index of window id {window_id} to {tab_to_focus_idx}')
        active_tab_local_idx = tab_to_focus_idx
        
        index += 1
        
    logging.info("Finished processing all books. You can now close the dedicated Chrome window.")

if __name__ == "__main__":
    if sys.platform == "darwin":
        print("---")
        print("macOS users: This script requires Accessibility permissions for Google Chrome.")
        print("Go to System Settings > Privacy & Security > Accessibility, and ensure your terminal/editor is enabled.")
        print("---")
    
    open_preview_links() 