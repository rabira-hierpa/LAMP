"""
Task management for the locust processing package.
"""

import time
import logging
from queue import Queue
from threading import Thread, Lock
from typing import Dict, List, Optional, Tuple, Any, Callable

from ..config import MAX_CONCURRENT_TASKS, MAX_RETRIES, RETRY_DELAY_SECONDS


class TaskManager:
    """
    Manager for Earth Engine export tasks with concurrent execution and retry handling.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_TASKS,
                 max_retries: int = MAX_RETRIES,
                 retry_delay: int = RETRY_DELAY_SECONDS):
        """
        Initialize the task manager.

        Args:
            max_concurrent: Maximum number of concurrent tasks
            max_retries: Maximum number of retry attempts for failed tasks
            retry_delay: Delay in seconds between retries
        """
        self.task_queue = Queue()
        self.active_tasks = {}  # task_id -> (task, description, attempts)
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.lock = Lock()
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.total_count = 0
        self.running = True

        # Start worker threads
        self.task_monitor = Thread(target=self._monitor_tasks)
        self.task_monitor.daemon = True
        self.task_monitor.start()

    def add_task(self, task_tuple: Optional[Tuple[Any, str]]) -> None:
        """
        Add a task to the queue.

        Args:
            task_tuple: Tuple of (task, description), or None if task should be skipped
        """
        if task_tuple is None:
            with self.lock:
                self.skipped_count += 1
                self.total_count += 1
            return

        task, description = task_tuple
        with self.lock:
            self.total_count += 1

        # (task, description, attempts)
        self.task_queue.put((task, description, 0))
        logging.info(f"➕ Added task to queue: {description}")

    def _start_task(self, task: Any, description: str, attempts: int) -> None:
        """
        Start a task and add it to active tasks.

        Args:
            task: Task to start
            description: Task description
            attempts: Number of previous attempts
        """
        task_id = f"{description}_{attempts}"
        with self.lock:
            if task_id in self.active_tasks:
                logging.warning(f"⚠️ Task {task_id} already in active tasks!")
                return

            self.active_tasks[task_id] = (task, description, attempts)

        try:
            task.start()
            logging.info(
                f"▶️ Started task: {description} (attempt {attempts+1})")
        except Exception as e:
            logging.error(f"❌ Error starting task {description}: {str(e)}")
            with self.lock:
                self.failed_count += 1
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

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
                        task, description, attempts = self.active_tasks[task_id]

                    try:
                        status = task.status()
                        state = status['state']

                        if state == 'COMPLETED':
                            logging.info(f"✅ Task completed: {description}")
                            with self.lock:
                                self.completed_count += 1
                                task_ids_to_remove.append(task_id)
                        elif state == 'FAILED':
                            logging.warning(
                                f"❌ Task failed: {description} - {status.get('error_message', 'Unknown error')}")
                            with self.lock:
                                task_ids_to_remove.append(task_id)

                            # Add to retry queue if not exceeded max retries
                            if attempts < self.max_retries:
                                logging.info(
                                    f"🔄 Scheduling retry for task: {description} (attempt {attempts+1})")
                                # Wait before retrying
                                time.sleep(self.retry_delay)
                                self.task_queue.put(
                                    (task, description, attempts + 1))
                            else:
                                logging.error(
                                    f"🛑 Task failed after {self.max_retries} attempts: {description}")
                                with self.lock:
                                    self.failed_count += 1
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

                # Start new tasks in batch
                with self.lock:
                    active_count = len(self.active_tasks)
                    room_for_tasks = self.max_concurrent - active_count

                # Add new tasks in larger batches
                if room_for_tasks > 0:
                    # Start multiple tasks (up to 20) in quick succession
                    batch_size = min(room_for_tasks, 20)
                    tasks_added = 0

                    for _ in range(batch_size):
                        if self.task_queue.empty():
                            break

                        try:
                            task, description, attempts = self.task_queue.get(
                                block=False)
                            self._start_task(task, description, attempts)
                            tasks_added += 1
                        except Exception as e:
                            logging.error(f"Error starting new task: {str(e)}")

                    if tasks_added > 0:
                        logging.info(
                            f"🚀 Started {tasks_added} new tasks in batch")

                    # Small sleep before starting next batch of tasks to avoid rate limits
                    if tasks_added > 0 and not self.task_queue.empty():
                        # Just a small pause to avoid hitting rate limits
                        time.sleep(1)

                # Print status every minute
                with self.lock:
                    completion_percentage = 0
                    if self.total_count > 0:
                        completion_percentage = (
                            self.completed_count / self.total_count) * 100

                    logging.info(f"📊 Status: {len(self.active_tasks)} active, {self.task_queue.qsize()} queued, "
                                 f"{self.completed_count} completed ({completion_percentage:.1f}%), "
                                 f"{self.failed_count} failed, {self.skipped_count} skipped, "
                                 f"{self.total_count} total")

                # Sleep shorter times between monitoring cycles
                time.sleep(15)  # Check twice as often

            except Exception as e:
                logging.error(f"💥 Error in task monitor: {str(e)}")
                time.sleep(10)  # Sleep a bit before retrying

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
        return self.completed_count, self.failed_count, self.skipped_count

    def shutdown(self) -> None:
        """Shutdown the task manager."""
        self.running = False
        logging.info("🛑 Shutting down task manager...")
        if self.task_monitor.is_alive():
            self.task_monitor.join(timeout=60)
