"""
Progress tracking functionality for the locust processing package.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Set, Tuple, Any, Optional


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
            logging.info(f"No progress file found at {progress_file}")
            return set(), 0, 0, 0

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
