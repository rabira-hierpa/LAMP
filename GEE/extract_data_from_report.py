import pandas as pd
import time
import os


def read_large_csv(file_path, required_columns):
    """
    Read specific columns from a large CSV file efficiently.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    required_columns : list
        List of column names to extract

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the required columns
    """
    start_time = time.time()

    # Get file size in MB
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Reading file: {file_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Extracting columns: {required_columns}")

    # Read only the required columns
    df = pd.read_csv(file_path, usecols=required_columns)

    print(f"Original number of rows: {len(df)}")
    # Filter out rows where 'Obs Date' is less than the year 2000
    df = df[df['Year'] >= 2015]
    print(f"Number of rows after filtering Obs Date after 2015: {len(df)}")

    # Format the date to YYYY-MM-DD
    # Use expand=True to get a DataFrame instead of a Series of lists
    date_time_parts = df['Obs Date'].str.split(' ', expand=True)
    date_parts = date_time_parts[0].str.split('/', expand=True)

    # Now we have a DataFrame with columns for month, day, year
    month = date_parts[0].str.zfill(2)  # Pad with leading zeros
    day = date_parts[1].str.zfill(2)    # Pad with leading zeros
    year = date_parts[2]                # Year (might contain extra characters)

    # Extract only the year part in case there are additional characters
    year = year.str.extract(r'(\d{4})', expand=False)

    # Create formatted date YYYY-MM-DD
    df['formatted_date'] = year + '-' + month + '-' + day

    print("Date formatting example:")
    print(df[['Obs Date', 'formatted_date']].head())

    # Sort by formatted_date
    df = df.sort_values(by='formatted_date', ascending=True)

    # If 'index' is not in required columns, add it
    if 'index' not in required_columns:
        df['index'] = df.index

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(
        f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

    return df


if __name__ == "__main__":
    # Path to your large CSV file
    file_path = "/Users/rz/Msc/Data/FAO_archival_data.csv"

    # Columns to extract from the FAO locust report
    required_columns = ['OBJECTID', 'Obs Date', 'Year', 'Locust Presence',
                        'Latitude', 'Longitude']

    # Read the required columns
    df = read_large_csv(file_path, required_columns)

    # Preview the data
    print("\nData Preview:")
    print(df.head())

    # # Save the extracted data to a new CSV file if needed
    output_path = "/Users/rz/Msc/Data/FAO_archival_data_extracted_2015.csv"
    df.to_csv(output_path, index=False)
    print(f"\nExtracted data saved to: {output_path}")
