"""
Test script to demonstrate task progress tracking.
"""

import os
import time
import logging
import json
from locust_processing.task_management.task_manager import TaskManager

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


def main():
    # Create a progress file path
    progress_file = os.path.join(
        os.path.dirname(__file__), "test_progress.json")

    # Initialize task manager with progress tracking
    task_manager = TaskManager(
        max_concurrent=3,
        max_retries=1,
        retry_delay=1,
        progress_file=progress_file
    )

    # Create some test tasks (mix of success and failure)
    tasks = [
        (MockTask(f"task_{i}", success=(i % 4 != 0)), f"Task {i}")
        for i in range(10)
    ]

    # Add tasks to the manager
    task_manager.add_tasks_batch(tasks)

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
        else:
            logging.error(f"Progress file not found: {progress_file}")

    finally:
        # Shut down the task manager
        task_manager.shutdown()
        logging.info("Test completed")


if __name__ == "__main__":
    main()
