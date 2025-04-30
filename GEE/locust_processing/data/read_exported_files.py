"""
Utility script to read exported files from Google Drive and track their progress.
"""

import os
import re
from datetime import datetime
import json
from typing import Dict, List, Optional, Set
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import logging

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def setup_logging(log_file: str = 'locust_export_progress.log') -> logging.Logger:
    """Configure logging to both console and file."""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


def get_google_drive_service():
    """Get Google Drive service with proper authentication."""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def get_folder_id(service, folder_name: str) -> Optional[str]:
    """Get the ID of a folder by its name."""
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
        spaces='drive',
        fields='files(id, name)'
    ).execute()

    items = results.get('files', [])
    if not items:
        return None
    return items[0]['id']


def list_files_in_folder(service, folder_id: str) -> List[Dict]:
    """List all files in a Google Drive folder."""
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        spaces='drive',
        fields='files(id, name, createdTime)'
    ).execute()

    return results.get('files', [])


def extract_file_info(filename: str) -> Optional[Dict]:
    """Extract date and index information from filename."""
    # Expected format: locust_YYYY-MM-DD_label_X_idx_Y
    pattern = r'locust_(\d{4}-\d{2}-\d{2})_label_(\d)_idx_(\d+)'
    match = re.match(pattern, filename)

    if not match:
        return None

    date_str, label, index = match.groups()
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        return {
            'date': date_str,
            'label': int(label),
            'index': int(index),
            'filename': filename
        }
    except ValueError:
        return None


def save_progress(progress_file: str, processed_files: Set[str],
                  total_files: int, processed_count: int) -> None:
    """Save progress to a JSON file."""
    progress = {
        'processed_files': list(processed_files),
        'total_files': total_files,
        'processed_count': processed_count,
        'last_updated': datetime.now().isoformat()
    }

    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def load_progress(progress_file: str) -> Dict:
    """Load progress from a JSON file."""
    if not os.path.exists(progress_file):
        return {
            'processed_files': [],
            'total_files': 0,
            'processed_count': 0,
            'last_updated': None
        }

    with open(progress_file, 'r') as f:
        return json.load(f)


def main():
    """Main function to read exported files and track progress."""
    logger = setup_logging()
    logger.info("Starting to read exported files from Google Drive")

    # Get Google Drive service
    try:
        service = get_google_drive_service()
        logger.info("Successfully authenticated with Google Drive")
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Drive: {str(e)}")
        return

    # Get folder ID
    folder_name = 'Locust_Exported_Images'
    folder_id = get_folder_id(service, folder_name)
    if not folder_id:
        logger.error(f"Could not find folder: {folder_name}")
        return

    # List files in folder
    files = list_files_in_folder(service, folder_id)
    logger.info(f"Found {len(files)} files in folder {folder_name}")

    # Load existing progress
    progress_file = 'locust_export_progress.json'
    progress = load_progress(progress_file)
    processed_files = set(progress['processed_files'])

    # Process files
    new_files = []
    for file in files:
        file_info = extract_file_info(file['name'])
        if file_info and file['name'] not in processed_files:
            new_files.append(file_info)
            processed_files.add(file['name'])

    # Update progress
    if new_files:
        logger.info(f"Found {len(new_files)} new files to process")
        save_progress(progress_file, processed_files,
                      len(files), len(processed_files))
        logger.info(f"Updated progress saved to {progress_file}")
    else:
        logger.info("No new files found to process")


if __name__ == "__main__":
    main()
