# Manual Quote Cleanup Tool

## Overview

The Manual Quote Cleanup Tool is a Flask-based web application designed to help manually review and correct text extracted from Google Books pages. It provides a user-friendly interface for comparing original extracted text with automatically cleaned versions and making manual corrections where needed.

## Input Data Structure

### Primary Input File

- **File**: `unquoted_google_books_text.csv`
- **Encoding**: UTF-8
- **Format**: CSV with the following key columns:
  - `LIKES`: Number of likes for the quote
  - `AUTHOR`: Author name
  - `TITLE`: Book title
  - `TAGS`: Associated tags
  - `language_or_country`: Language/country information
  - `source_method`: How the data was sourced
  - `publication_date`: Book publication date
  - `genres`: Book genres
  - `preview_link`: Google Books preview URL
  - `Filename`: Associated screenshot filename
  - `QUOTE`: Original quote text
  - `Google Books Page Text`: Raw extracted text from Google Books page
  - `matched_quotes_list`: List of matched quotes found in the page
  - `Google Books Page Text_unquoted`: Automatically cleaned version of the page text

### Progress Tracking Files

- **`cleanup_review_progress.txt`**: Tracks the current review index (which row is being reviewed)
- **`manual_review_progress.txt`**: Additional progress tracking

## Output Data Structure

### Primary Output File

- **File**: `unquoted_text_manual_review.csv`
- **Format**: Same structure as input CSV, but with manually corrected `Google Books Page Text_unquoted` column
- **Creation**: File is created when the first manual correction is saved

## How It Works

### 1. Data Processing Pipeline

1. **Input**: Reads from `unquoted_google_books_text.csv` (or existing corrected file)
2. **Filtering**: Identifies rows where `Google Books Page Text` differs from `Google Books Page Text_unquoted`
3. **Review Queue**: Creates a queue of items needing manual review
4. **Progress Tracking**: Maintains position in the review queue

### 2. Text Comparison & Highlighting

The tool uses advanced diff algorithms to highlight changes:

- **Red Strikethrough**: Text that was automatically removed
- **Gold Highlighting**: Words immediately before and after substantial deletions (context markers)
- **Smart Filtering**: Ignores whitespace-only changes to focus on meaningful content differences

### 3. Manual Review Interface

- **Editable Area**: Click-to-edit text display with rich formatting
- **Context Information**: Shows book title, author, and progress
- **Visual Feedback**: Clear indication of what was changed and why

### 4. Keyboard Shortcut (NEW)

- **`*`**: Save current edits and advance to next item
- **Visual Feedback**: Button briefly changes color when shortcut is used
- **Smart Prevention**: Prevents the `*` character from being typed when used as shortcut

## Usage Instructions

### Starting the Application

```bash
cd "Neutral Snippet Dataset"
python manual_quote_cleanup.py
```

### Accessing the Interface

- Open browser to: `http://127.0.0.1:5002`
- The tool automatically loads the next item needing review

### Review Process

1. **Examine** the highlighted text differences
2. **Edit** the text in the editable area as needed
3. **Save** using the button or keyboard shortcut:
   - Click "Save and Go to Next"
   - Press `*`
4. **Repeat** until all items are reviewed

### Completion

When all items are reviewed, the tool displays "All Done!" message.

## Technical Details

### Text Processing Algorithm

```python
def create_final_editable_html(original, cleaned):
    # Uses difflib.SequenceMatcher for token-level comparison
    # Identifies substantial deletion blocks (ignoring whitespace)
    # Highlights context words (before/after deletions)
    # Returns HTML with visual markup
```

### Data Flow

1. **Load**: CSV → Pandas DataFrame
2. **Filter**: Find rows with differences
3. **Process**: Generate diff HTML
4. **Display**: Render in web interface
5. **Save**: Update DataFrame → CSV

### File Management

- **Backup Strategy**: Original file remains untouched
- **Incremental Saves**: Each correction immediately saved
- **Progress Persistence**: Review position saved after each item

## Error Handling

- **Missing Files**: Graceful fallback with error messages
- **Invalid Indices**: Bounds checking for array access
- **Encoding Issues**: UTF-8 handling throughout
- **Browser Compatibility**: Modern JavaScript features with fallbacks

## Performance Considerations

- **Memory Efficient**: Processes one row at a time
- **Fast Diff**: Optimized token-based comparison
- **Responsive UI**: Immediate feedback on user actions
- **Progress Tracking**: Resume from any point

## Common Use Cases

1. **Quote Extraction Cleanup**: Remove unwanted text from book page extractions
2. **OCR Correction**: Fix text recognition errors
3. **Format Standardization**: Ensure consistent text formatting
4. **Quality Assurance**: Manual verification of automated cleaning

## Dependencies

- **Flask**: Web framework
- **Pandas**: Data manipulation
- **difflib**: Text comparison
- **html**: HTML escaping
- **re**: Regular expressions
- **os**: File operations
- **logging**: Activity logging
