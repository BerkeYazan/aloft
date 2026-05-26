import sqlite3
import pandas as pd
import os

def sqlite_to_csv():
    # Define file paths
    data_dir = os.path.join(os.path.dirname(__file__), 'Data')
    sqlite_file = os.path.join(data_dir, 'quotes.sqlite')
    csv_file = os.path.join(data_dir, 'quotes-350k.csv')
    
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_file)
    
    # Query to get the first 350,000 quotes
    query = "SELECT * FROM quotes LIMIT 380000"
    
    # Use pandas to read the query results and write to CSV
    df = pd.read_sql_query(query, conn)
    
    # Convert to CSV
    df.to_csv(csv_file, index=False)
    
    # Close the connection
    conn.close()
    
    print(f"Conversion complete. CSV file saved at: {csv_file}")
    print(f"Number of rows exported: {len(df)}")

if __name__ == "__main__":
    sqlite_to_csv() 