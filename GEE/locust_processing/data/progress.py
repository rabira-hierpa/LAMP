"""
Progress tracking functionality for the locust processing package.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from ..config import EXPORT_FOLDER
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def save_progress(progress_file: str,
                  processed_indices: List[int],
                  completed_count: int,
                  failed_count: int,
                  skipped_count: int) -> bool:
    """
    Save progress to a JSON file.

    Args:
        progress_file: Path to the progress file
        processed_indices: List of indices that have been processed
        completed_count: Number of completed tasks
        failed_count: Number of failed tasks
        skipped_count: Number of skipped tasks

    Returns:
        bool: True if progress was saved successfully, False otherwise
    """
    try:
        progress_data = {
            'processed_indices': processed_indices,
            'completed_count': completed_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'timestamp': datetime.datetime.now().isoformat()
        }

        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)

        logging.info(f"Progress saved to {progress_file}")
        return True

    except Exception as e:
        logging.error(f"Error saving progress: {str(e)}")
        return False


def load_progress(progress_file: str) -> Tuple[Set[int], int, int, int]:
    """
    Load progress from a JSON file.

    Args:
        progress_file: Path to the progress file

    Returns:
        Tuple of (processed_indices_set, completed_count, failed_count, skipped_count)
    """
    try:
        if not os.path.exists(progress_file):
            logging.info(
                f"No progress file found at {progress_file}, creating new one")
            with open(progress_file, 'w') as f:
                progress_data = {
                    'processed_indices': [],
                    'completed_count': 0,
                    'failed_count': 0,
                    'skipped_count': 0,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                json.dump(progress_data, f)
            logging.info(f"🆕 New progress file created at {progress_file}")

        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        processed_indices = set(progress_data.get('processed_indices', []))
        completed_count = progress_data.get('completed_count', 0)
        failed_count = progress_data.get('failed_count', 0)
        skipped_count = progress_data.get('skipped_count', 0)
        timestamp = progress_data.get('timestamp', 'unknown')

        logging.info(f"Loaded progress from {progress_file} (saved at {timestamp}): {len(processed_indices)} processed, "
                     f"{completed_count} completed, {failed_count} failed, {skipped_count} skipped")

        return processed_indices, completed_count, failed_count, skipped_count

    except Exception as e:
        logging.error(f"Error loading progress: {str(e)}")
        return set(), 0, 0, 0


def get_progress_summary(processed_indices: Set[int],
                         completed_count: int,
                         failed_count: int,
                         skipped_count: int) -> Dict[str, Any]:
    """
    Get a summary of the current progress.

    Args:
        processed_indices: Set of indices that have been processed
        completed_count: Number of completed tasks
        failed_count: Number of failed tasks
        skipped_count: Number of skipped tasks

    Returns:
        Dict: Summary dictionary with progress metrics
    """
    total_processed = len(processed_indices)
    total_count = completed_count + failed_count + skipped_count

    return {
        'total_processed': total_processed,
        'completed_count': completed_count,
        'failed_count': failed_count,
        'skipped_count': skipped_count,
        'total_count': total_count,
        'completion_percentage': 100 * completed_count / total_count if total_count > 0 else 0,
        'timestamp': datetime.datetime.now().isoformat()
    }


def setup_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens browser for OAuth
    return GoogleDrive(gauth)


def update_progress_file(progress_file: str):

    # Mount Google Drive to access the exported data
    drive = setup_drive()

    file_list = drive.ListFile(
        {'q': f"'{EXPORT_FOLDER}' in parents"}).GetList()
    tif_files = [f for f in file_list if f['title'].endswith('.tif')]

    progress_data = {
        'processed_indices': [],
        'completed_count': 0,
        'failed_count': 0,
        'skipped_count': 0,
        'timestamp': datetime.datetime.now().isoformat(),
        'presence_count': 0,
        'absence_count': 0
    }

    file_names = os.path.splitext(tif_files[0])[0]
    print(f"File name", file_names.split('_'))
    presence_count = 0
    absence_count = 0
    for file in files:
        file_path = os.path.join(data_path, file)
        # Extract file names
        file_name = os.path.splitext(file)[0]
        file_idx = file_name.split('_')[-1]
        if 'label_1' in file_name:
            presence_count += 1
        else:
            absence_count += 1
        progress_data['processed_indices'].append(int(file_idx))
    print(f"Found {len(files)} GeoTIFF files.")
    print(f"Processed indices: {len(progress_data['processed_indices'])}")
    progress_data['completed_count'] = len(progress_data['processed_indices'])
    progress_data['failed_count'] = 0
    progress_data['skipped_count'] = 0
    progress_data['presence_count'] = presence_count
    progress_data['absence_count'] = absence_count
    print(f"Presence count: {presence_count}")
    print(f"Absence count: {absence_count}")

    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
        print(f"Progress file {progress_file} updated successfully!")
    except Exception as e:
        print(f"Error writing progress file: {e}")
