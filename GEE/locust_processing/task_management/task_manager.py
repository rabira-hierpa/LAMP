"""
Task management for the locust processing package.
"""

import time
import logging
import json
import os
from queue import Queue
from threading import Thread, Lock
from typing import Dict, List, Optional, Tuple, Any, Callable, Set

from ..config import MAX_CONCURRENT_TASKS, MAX_RETRIES, RETRY_DELAY_SECONDS, DEFAULT_PROGRESS_FILE


def save_progress(progress_file: str,
                  processed_indices: List[int],
                  completed_count: int,
                  failed_count: int,
                  skipped_count: int) -> None:
    """
    Save progress to a JSON file.

    Args:
        progress_file: Path to the progress file
        processed_indices: List of OBJECT_IDs that have been processed
        completed_count: Number of completed tasks
        failed_count: Number of failed tasks
        skipped_count: Number of skipped tasks
    """
    if not progress_file:
        return

    progress_data = {
        "processed_indices": processed_indices,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "timestamp": datetime.datetime.now().isoformat() if 'datetime' in globals() else time.time()
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    # Save progress to file
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
        logging.info(f"Progress saved to {progress_file}")
    except Exception as e:
        logging.error(f"Error saving progress to {progress_file}: {str(e)}")


class TaskManager:
    """
    Manager for Earth Engine export tasks with concurrent execution and retry handling.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_TASKS,
                 max_retries: int = MAX_RETRIES,
                 retry_delay: int = RETRY_DELAY_SECONDS,
                 progress_file: str = DEFAULT_PROGRESS_FILE):
        """
        Initialize the task manager.

        Args:
            max_concurrent: Maximum number of concurrent tasks
            max_retries: Maximum number of retry attempts for failed tasks
            retry_delay: Delay in seconds between retries
            progress_file: Path to save progress information (if None, progress is not saved)
        """
        self.task_queue = Queue()
        # task_id -> (task, description, attempts, object_id)
        self.active_tasks = {}
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.progress_file = progress_file
        self.lock = Lock()
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.total_count = 0
        self.running = True
        self.completed_descriptions = []  # Track all completed task descriptions
        self.processed_object_ids = set()  # Track processed OBJECT_IDs

        # Start worker threads
        self.task_monitor = Thread(target=self._monitor_tasks)
        self.task_monitor.daemon = True
        self.task_monitor.start()

    def add_task(self, task: Any, description: str, object_id: Optional[int] = None) -> None:
        """
        Add a task to the queue and try to start it immediately if possible.

        Args:
            task: The task to add
            description: Description of the task
            object_id: Optional OBJECT_ID associated with this task
        """
        with self.lock:
            self.total_count += 1
            if object_id is not None:
                self.processed_object_ids.add(object_id)

        # (task, description, attempts, object_id)
        self.task_queue.put((task, description, 0, object_id))
        logging.info(f"➕ Added task to queue: {description}")

        # Try to start the task immediately if we have room
        self.batch_start_tasks()

    def add_tasks_batch(self, task_tuples: List[Tuple[Any, str, Optional[int]]]) -> int:
        """
        Add multiple tasks to the queue at once and try to start them immediately.

        Args:
            task_tuples: List of (task, description, object_id) tuples

        Returns:
            int: Number of tasks added
        """
        with self.lock:
            self.total_count += len(task_tuples)

        # Add all tasks to the queue
        count = 0
        for task_tuple in task_tuples:
            if len(task_tuple) == 2:
                task, description = task_tuple
                object_id = None
            else:
                task, description, object_id = task_tuple

            if object_id is not None:
                with self.lock:
                    self.processed_object_ids.add(object_id)

            # (task, description, attempts, object_id)
            self.task_queue.put((task, description, 0, object_id))
            count += 1

        logging.info(f"➕ Added {count} tasks to queue in batch")

        # Try to start as many tasks as possible immediately
        self.batch_start_tasks()

        return count

    def batch_start_tasks(self, max_to_start: int = None) -> int:
        """
        Start multiple tasks from the queue in a single batch.

        Args:
            max_to_start: Maximum number of tasks to start in this batch. If None, uses max_concurrent.

        Returns:
            int: Number of tasks actually started
        """
        tasks_started = 0
        tasks_to_start = []

        # Get tasks from queue (up to max_to_start)
        with self.lock:
            active_count = len(self.active_tasks)
            room_for_tasks = self.max_concurrent - active_count
            if max_to_start is not None:
                room_for_tasks = min(room_for_tasks, max_to_start)

            # If no room for tasks, return early
            if room_for_tasks <= 0:
                logging.debug(
                    f"No room for new tasks (active: {active_count}/{self.max_concurrent})")
                return 0

            # Get tasks from queue
            for _ in range(room_for_tasks):
                if self.task_queue.empty():
                    break
                try:
                    task_tuple = self.task_queue.get(block=False)
                    tasks_to_start.append(task_tuple)
                except Exception as e:
                    logging.debug(f"Error getting task from queue: {str(e)}")
                    break

        # Start tasks outside the lock to avoid holding it for too long
        for task, description, attempts, object_id in tasks_to_start:
            task_id = f"{description}_{attempts}"

            # Add to active tasks
            with self.lock:
                if task_id in self.active_tasks:
                    logging.warning(
                        f"⚠️ Task {task_id} already in active tasks!")
                    continue
                self.active_tasks[task_id] = (
                    task, description, attempts, object_id)

            # Start the task
            try:
                task.start()
                tasks_started += 1
                logging.info(
                    f"▶️ Started task: {description} (attempt {attempts+1})")
            except Exception as e:
                logging.error(f"❌ Error starting task {description}: {str(e)}")
                with self.lock:
                    self.failed_count += 1
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]

        if tasks_started > 0:
            logging.info(
                f"🚀 Started {tasks_started} tasks in batch (active: {len(self.active_tasks)}/{self.max_concurrent})")

        return tasks_started

    def _monitor_tasks(self) -> None:
        """Monitor running tasks and start new ones as needed."""
        while self.running:
            try:
                # Process any completed tasks
                task_ids_to_remove = []

                with self.lock:
                    active_task_ids = list(self.active_tasks.keys())

                for task_id in active_task_ids:
                    with self.lock:
                        if task_id not in self.active_tasks:
                            continue
                        task, description, attempts, object_id = self.active_tasks[task_id]

                    try:
                        state = task.status()['state']
                        if state == 'COMPLETED':
                            logging.info(f"✅ Task completed: {description}")
                            with self.lock:
                                self.completed_count += 1
                                self.completed_descriptions.append(description)
                                task_ids_to_remove.append(task_id)

                            # Save progress after task completion
                            save_progress(
                                self.progress_file,
                                list(self.processed_object_ids),
                                self.completed_count,
                                self.failed_count,
                                self.skipped_count
                            )

                        elif state == 'FAILED':
                            error_msg = task.status().get('error_message', 'Unknown error')
                            logging.warning(
                                f"❌ Task failed: {description} - {error_msg}")
                            with self.lock:
                                task_ids_to_remove.append(task_id)

                            # Add to retry queue if not exceeded max retries
                            if attempts < self.max_retries:
                                logging.info(
                                    f"🔄 Scheduling retry for task: {description} (attempt {attempts+1})")
                                # Wait before retrying
                                time.sleep(self.retry_delay)
                                self.task_queue.put(
                                    (task, description, attempts + 1, object_id))
                            else:
                                logging.error(
                                    f"🛑 Task failed after {self.max_retries} attempts: {description}")
                                with self.lock:
                                    self.failed_count += 1
                                    # Also save progress when a task fails permanently
                                    save_progress(
                                        self.progress_file,
                                        list(self.processed_object_ids),
                                        self.completed_count,
                                        self.failed_count,
                                        self.skipped_count
                                    )
                        elif state in ['CANCELLED', 'CANCEL_REQUESTED']:
                            logging.warning(
                                f"⚠️ Task cancelled: {description}")
                            with self.lock:
                                self.failed_count += 1
                                task_ids_to_remove.append(task_id)
                    except Exception as e:
                        logging.error(
                            f"Error monitoring task {description}: {str(e)}")
                        with self.lock:
                            task_ids_to_remove.append(task_id)

                # Remove completed/failed tasks
                with self.lock:
                    for task_id in task_ids_to_remove:
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]

                # Calculate available slots and queue size
                with self.lock:
                    active_count = len(self.active_tasks)
                    available_slots = self.max_concurrent - active_count
                    queue_size = self.task_queue.qsize()

                # Start as many tasks as possible, up to max_concurrent
                if available_slots > 0 and queue_size > 0:
                    # Use the full available slots to start tasks
                    batch_size = min(available_slots, queue_size)
                    logging.info(
                        f"Starting up to {batch_size} new tasks (active: {active_count}, available: {available_slots}, queued: {queue_size})")
                    started = self.batch_start_tasks()

                    # If we started tasks and there are more in the queue and still room, start another batch immediately
                    if started > 0 and not self.task_queue.empty() and (len(self.active_tasks) < self.max_concurrent):
                        continue

                # Print status every cycle
                with self.lock:
                    completion_percentage = 0
                    if self.total_count > 0:
                        completion_percentage = (
                            self.completed_count / self.total_count) * 100

                    queue_size = self.task_queue.qsize()
                    active_count = len(self.active_tasks)

                    logging.info(f"📊 Status: {active_count}/{self.max_concurrent} active, {queue_size} queued, "
                                 f"{self.completed_count} completed ({completion_percentage:.1f}%), "
                                 f"{self.failed_count} failed, {self.skipped_count} skipped, "
                                 f"{self.total_count} total")

                # Small sleep to avoid busy waiting
                time.sleep(0.1)

            except Exception as e:
                logging.error(f"💥Error in task monitor: {str(e)}")
                time.sleep(1)  # Sleep a bit before retrying

    def wait_until_complete(self) -> Tuple[int, int, int]:
        """
        Wait until all tasks are completed.

        Returns:
            Tuple of (completed_count, failed_count, skipped_count)
        """
        while self.running and (not self.task_queue.empty() or len(self.active_tasks) > 0):
            time.sleep(10)

        logging.info(
            f"🏁 All tasks completed. Results: ✅ {self.completed_count} completed, ❌ {self.failed_count} failed, ⏭️ {self.skipped_count} skipped")

        # Final save of progress file
        save_progress(
            self.progress_file,
            list(self.processed_object_ids),
            self.completed_count,
            self.failed_count,
            self.skipped_count
        )

        return self.completed_count, self.failed_count, self.skipped_count

    def shutdown(self) -> None:
        """Shutdown the task manager."""
        self.running = False
        logging.info("🛑 Shutting down task manager...")
        if self.task_monitor.is_alive():
            self.task_monitor.join(timeout=60)

        # Final save of progress file
        save_progress(
            self.progress_file,
            list(self.processed_object_ids),
            self.completed_count,
            self.failed_count,
            self.skipped_count
        )

    def increment_skipped(self, object_id: Optional[int] = None) -> None:
        """
        Increment the skipped count in a thread-safe manner.

        Args:
            object_id: Optional OBJECT_ID to mark as processed
        """
        with self.lock:
            self.skipped_count += 1
            self.total_count += 1
            if object_id is not None:
                self.processed_object_ids.add(object_id)
            logging.info(f"Skipped count: {self.skipped_count}")
