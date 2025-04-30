# Locust Processing Package

This package provides utilities for processing locust data, including Google Drive integration for managing exported files.

## Installation

1. Clone this repository
2. Install the required Python packages:

```bash
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

## Google Drive Integration Setup

To use the Google Drive integration features:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API
4. Create OAuth 2.0 credentials
5. Download the credentials and save them as `credentials.json` in the same directory as the script

## Usage

The package provides various utilities for processing locust data. For Google Drive integration:

```python
from locust_processing.utils.read_exported_files import main

# Run the script to read exported files from Google Drive
main()
```

The script will:

- Create a `token.json` file after the first successful authentication
- Create a `locust_export_progress.json` file to track processed files
- Create a `locust_export_progress.log` file for logging

## Progress Tracking

The progress file (`locust_export_progress.json`) contains:

- List of processed files
- Total number of files
- Number of processed files
- Last update timestamp
