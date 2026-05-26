# 00 Setup

This stage prepares your machine to run the pipeline. There is no code to run
here, only setup steps.

## Steps

1. Install Python. The pipeline was developed on Python 3.11. Other versions
   may work but were not tested.

2. Create a virtual environment and install the dependencies:

   ```
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up credentials. Two external services are used:

   - The Google Books API, used in stage 01 to look up publication years and
     other book metadata.
   - Google Cloud Vision, used in stage 02 to read text from screenshots of
     Google Books pages.

   Copy `.env.example` from the repository root to a new file called `.env`
   and fill in your own values. The `.env` file is ignored by git and must
   never be committed.

## Notes

- Run every script from the repository root. See the main README for the
  exact command form.
- `requirements.txt` lists the packages used by the pipeline.
