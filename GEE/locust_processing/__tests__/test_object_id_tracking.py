"""
Test script to demonstrate OBJECT_ID tracking in the TaskManager.
"""

import os
import time
import logging
import json
import random
from typing import List

from locust_processing.task_management.task_manager import TaskManager
from locust_processing.config import DEFAULT_PROGRESS_FILE

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Mock Earth Engine task for testing


class MockTask:
    def __init__(self, task_id, success=True, delay=1):
        self.task_id = task_id
        self.success = success
        self.delay = delay
        self.started = False
        self.state = "READY"

    def start(self):
        self.started = True
        self.state = "RUNNING"
        logging.info(f"Started task {self.task_id}")

    def status(self):
        # Simulate task completion after delay
        if self.started:
            if self.state == "RUNNING":
                # Simulate task running for a bit
                time.sleep(self.delay)
                if self.success:
                    self.state = "COMPLETED"
                else:
                    self.state = "FAILED"

            return {"state": self.state,
                    "error_message": "Mock failure" if self.state == "FAILED" else None}
        return {"state": "READY"}


def generate_random_object_ids(count: int) -> List[int]:
    """Generate a list of random OBJECT_IDs for testing"""
    return [random.randint(10000, 99999) for _ in range(count)]


def main():
    # Create a progress file path
    progress_file = os.path.join(
        os.path.dirname(__file__), "test_object_ids.json")

    # Delete existing progress file if it exists
    if os.path.exists(progress_file):
        os.remove(progress_file)
        logging.info(f"Removed existing progress file: {progress_file}")

    # Initialize task manager with progress tracking
    task_manager = TaskManager(
        max_concurrent=3,
        max_retries=1,
        retry_delay=1,
        progress_file=progress_file
    )

    # Generate some random OBJECT_IDs
    object_ids = generate_random_object_ids(10)
    logging.info(f"Generated OBJECT_IDs: {object_ids}")

    # Create tasks with OBJECT_IDs
    tasks = [
        (MockTask(f"task_{i}", success=(i % 4 != 0)),
         f"Task {i}", object_ids[i])
        for i in range(10)
    ]

    # Add tasks to the manager
    task_manager.add_tasks_batch(tasks)

    # Process all tasks
    try:
        # Wait for completion
        logging.info("Waiting for tasks to complete...")
        completed, failed, skipped = task_manager.wait_until_complete()

        # Print final results
        logging.info(
            f"Final results: {completed} completed, {failed} failed, {skipped} skipped")

        # Read the progress file and display contents
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                logging.info(
                    f"Progress file contents: {json.dumps(progress_data, indent=2)}")

                # Verify that all OBJECT_IDs were saved
                saved_ids = set(map(int, progress_data['processed_indices']))
                original_ids = set(object_ids)
                if saved_ids == original_ids:
                    logging.info("✅ All OBJECT_IDs were correctly saved!")
                else:
                    missing_ids = original_ids - saved_ids
                    extra_ids = saved_ids - original_ids
                    if missing_ids:
                        logging.error(f"❌ Missing OBJECT_IDs: {missing_ids}")
                    if extra_ids:
                        logging.error(f"❌ Unexpected OBJECT_IDs: {extra_ids}")
        else:
            logging.error(f"Progress file not found: {progress_file}")

    finally:
        # Shut down the task manager
        task_manager.shutdown()
        logging.info("Test completed")


if __name__ == "__main__":
    main()
