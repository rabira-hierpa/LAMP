"""
Command-line interface for the locust processing package.
"""

import argparse
import logging
import time
from typing import Dict, List, Set, Optional, Any

import ee

from .utils.logging_utils import setup_logging
from .utils.ee_utils import initialize_ee, ensure_ee_initialized, add_cumulative_index
from .utils.geo_utils import get_ethiopia_boundary
from .processing.indices import set_region_boundary
from .processing.export import create_export_task, start_export_task
from .task_management.task_manager import TaskManager
from .data.progress import save_progress, load_progress
from .config import (
    COMMON_SCALE,
    COMMON_PROJECTION,
    FAO_REPORT_ASSET_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROGRESS_FILE,
    DEFAULT_LOG_FILE,
    MAX_CONCURRENT_TASKS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS
)


def process_single_feature(indexed_collection: ee.FeatureCollection,
                           idx: int,
                           processed_indices: Set[int],
                           task_manager: TaskManager) -> bool:
    """
    Process a single feature by index.

    Args:
        indexed_collection: FeatureCollection with index property
        idx: Index of the feature to process
        processed_indices: Set of already processed indices
        task_manager: TaskManager instance for task handling

    Returns:
        bool: True if the feature was processed successfully, False otherwise
    """
    if idx in processed_indices:
        logging.info(f"Skipping already processed feature index {idx}")
        return True

    try:
        # Filter the indexed collection by index
        feature_to_process = indexed_collection.filter(
            ee.Filter.eq('index', idx)).first()

        if feature_to_process.getInfo() is None:
            logging.warning(f"Feature with index {idx} not found. Skipping.")
            return False

        task_tuple = create_export_task(idx, feature_to_process)
        # Adds None if creation failed, handled by add_task
        task_manager.add_task(task_tuple)
        # Add to processed only if submitted or skipped by create_export_task
        processed_indices.add(idx)
        return True
    except ee.EEException as e:
        logging.error(
            f"Earth Engine error processing feature index {idx}: {e}")
        return False
    except Exception as e:
        logging.error(
            f"General error processing feature index {idx}: {str(e)}")
        return False


def process_in_test_mode(indexed_collection: ee.FeatureCollection,
                         args: argparse.Namespace) -> None:
    """
    Process a single feature in test mode.

    Args:
        indexed_collection: FeatureCollection with index property
        args: Command-line arguments
    """
    logging.info(f"--- Running in Test Mode (Index: {args.start_index}) ---")
    try:
        # Filter the indexed collection to get the single point
        single_point = indexed_collection.filter(
            ee.Filter.eq('index', args.start_index)).first()

        # Check if the feature was found
        if single_point.getInfo() is None:
            logging.error(
                f"Test mode: Feature with index {args.start_index} not found in the indexed collection.")
            return

        logging.info(
            f"Test mode: Processing feature with index {args.start_index}")
        logging.info(
            f"Test feature properties: {single_point.getInfo().get('properties', {})}")

        task_tuple = create_export_task(args.start_index, single_point)

        if task_tuple:
            task, description = task_tuple
            start_export_task(task, description)
            logging.info(f'Test export task started: {description}')

            # Wait for the task to complete or fail
            while task.status()['state'] in ('READY', 'RUNNING'):
                logging.info(
                    f"Test task status: {task.status()['state']} for {description}")
                time.sleep(10)

            final_status = task.status()
            logging.info(
                f"Test task '{description}' finished with state: {final_status['state']}")

            if final_status['state'] == 'COMPLETED':
                logging.info("Test export successful.")
            else:
                logging.error(
                    f"Test export failed: {final_status.get('error_message', 'Unknown error')}")
        else:
            logging.warning(
                f"Test export task for index {args.start_index} could not be created (check logs for reasons like missing data or invalid properties).")

    except ee.EEException as e:
        logging.error(
            f"Earth Engine error during test mode for index {args.start_index}: {e}")
    except Exception as e:
        logging.error(
            f"Unexpected error during test mode for index {args.start_index}: {e}")
    finally:
        logging.info("--- Exiting Test Mode ---")


def process_in_batch_mode(indexed_collection: ee.FeatureCollection,
                          args: argparse.Namespace,
                          processed_indices: Set[int],
                          initial_completed: int,
                          initial_failed: int,
                          initial_skipped: int) -> None:
    """
    Process features in batch mode.

    Args:
        indexed_collection: FeatureCollection with index property
        args: Command-line arguments
        processed_indices: Set of already processed indices
        initial_completed: Initial count of completed tasks
        initial_failed: Initial count of failed tasks
        initial_skipped: Initial count of skipped tasks
    """
    logging.info(f"--- Running in Batch Mode ---")

    # Initialize task manager for batch processing
    task_manager = TaskManager(
        max_concurrent=MAX_CONCURRENT_TASKS,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY_SECONDS
    )

    # Pass initial loaded counts to task manager
    task_manager.completed_count = initial_completed
    task_manager.failed_count = initial_failed
    task_manager.skipped_count = initial_skipped

    # Calculate total based on the range we intend to process
    feature_count = indexed_collection.size().getInfo()
    max_idx = args.start_index + args.max_features if args.max_features else feature_count
    # Ensure we don't exceed the available features
    max_idx = min(max_idx, feature_count)

    task_manager.total_count = max_idx - args.start_index

    try:
        # Process in batches to avoid memory issues
        logging.info(
            f"Starting batch processing from index {args.start_index} to {max_idx-1}")

        # Iterate through the required index range
        processed_in_run = 0
        for current_index in range(args.start_index, max_idx):
            success = process_single_feature(
                indexed_collection, current_index, processed_indices, task_manager)
            if success:  # Only count successfully submitted/skipped features towards progress
                processed_in_run += 1

            # Save progress periodically
            if processed_in_run > 0 and processed_in_run % 10 == 0:
                logging.info(
                    f"Processed {processed_in_run} features in this run. Saving progress...")
                save_progress(
                    args.progress_file,
                    list(processed_indices),
                    task_manager.completed_count,
                    task_manager.failed_count,
                    task_manager.skipped_count
                )

        # Wait for all tasks to complete
        logging.info(
            "All features submitted to queue. Waiting for tasks to complete...")
        final_completed, final_failed, final_skipped = task_manager.wait_until_complete()

        logging.info(
            f"Batch processing completed: {final_completed} successful, {final_failed} failed, {final_skipped} skipped")

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving progress...")
        # Save progress immediately on interrupt
        save_progress(
            args.progress_file,
            list(processed_indices),
            task_manager.completed_count,
            task_manager.failed_count,
            task_manager.skipped_count
        )
    except Exception as e:
        logging.error(f"Error in main batch processing loop: {str(e)}")
    finally:
        # Shutdown task manager
        logging.info("Shutting down task manager...")
        task_manager.shutdown()

        # Save final progress
        logging.info("Saving final progress...")
        save_progress(
            args.progress_file,
            list(processed_indices),
            task_manager.completed_count,
            task_manager.failed_count,
            task_manager.skipped_count
        )

        logging.info("Script completed. Final progress saved.")


def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Export locust data from Earth Engine')
    parser.add_argument('--test', action='store_true',
                        help='Run with a single test point')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of features to process in one batch')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Index to start processing from')
    parser.add_argument('--max-features', type=int, default=None,
                        help='Maximum number of features to process')
    parser.add_argument('--presence-only', action='store_true',
                        help='Process only presence points')
    parser.add_argument('--absence-only', action='store_true',
                        help='Process only absence points')
    parser.add_argument('--progress-file', type=str, default=DEFAULT_PROGRESS_FILE,
                        help='File to save/load progress')
    parser.add_argument('--log-file', type=str, default=DEFAULT_LOG_FILE,
                        help='Log file name')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_file)

    # Initialize Earth Engine
    initialize_ee()

    # Get Ethiopia boundary and set it for index calculations
    et_boundary = get_ethiopia_boundary()
    set_region_boundary(et_boundary)

    # Create global variables needed for calculation
    logging.info("Loading FAO locust data...")

    try:
        # Load FAO locust data
        locust_data = ee.FeatureCollection(FAO_REPORT_ASSET_ID)
        original_count = locust_data.size().getInfo()
        logging.info(
            f"Successfully loaded {original_count} features from {FAO_REPORT_ASSET_ID}")
    except ee.EEException as e:
        logging.error(
            f"Failed to load FeatureCollection: {FAO_REPORT_ASSET_ID}. Error: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return

    # Filter out features with null coordinates
    locust_data = locust_data.filter(ee.Filter.And(
        ee.Filter.neq('Longitude', None),
        ee.Filter.neq('Latitude', None)
    ))
    count_after_coord_filter = locust_data.size().getInfo()
    logging.info(
        f"Features after filtering null coordinates: {count_after_coord_filter} (removed {original_count - count_after_coord_filter})")

    # Filter out features with null 'Obs Date'
    locust_data = locust_data.filter(ee.Filter.neq('Obs Date', None))
    count_after_date_filter = locust_data.size().getInfo()
    logging.info(
        f"Features after filtering null 'Obs Date': {count_after_date_filter} (removed {count_after_coord_filter - count_after_date_filter})")

    # Filter out features with null 'Locust Presence'
    locust_data = locust_data.filter(ee.Filter.neq('Locust Presence', None))
    filtered_count = locust_data.size().getInfo()
    logging.info(
        f"Features after filtering null 'Locust Presence': {filtered_count} (removed {count_after_date_filter - filtered_count})")

    # Add geometry to features using Longitude and Latitude
    def add_geometry(feature):
        try:
            # Attempt to parse coordinates server-side
            lon = ee.Number.parse(feature.get('Longitude'))
            lat = ee.Number.parse(feature.get('Latitude'))
            # Return feature with geometry if parsing is successful
            return feature.setGeometry(ee.Geometry.Point([lon, lat]))
        except:
            # Return the feature without geometry if parsing fails
            return feature.setGeometry(None)  # Set geometry to null explicitly

    locust_data_with_geometry = locust_data.map(add_geometry)

    # Filter out features where geometry couldn't be added
    locust_data_with_geometry = locust_data_with_geometry.filter(
        ee.Filter.neq('geometry', None))
    count_after_geom_filter = locust_data_with_geometry.size().getInfo()
    logging.info(
        f"Features after adding geometry and filtering nulls: {count_after_geom_filter} (removed {filtered_count - count_after_geom_filter})")

    # Filter for presence and absence points based on the geometry-added collection
    if args.presence_only:
        logging.info("Processing only presence points")
        filtered_data = locust_data_with_geometry.filter(
            ee.Filter.eq('Locust Presence', 'PRESENT'))
    elif args.absence_only:
        logging.info("Processing only absence points")
        filtered_data = locust_data_with_geometry.filter(
            ee.Filter.eq('Locust Presence', 'ABSENT'))
    else:
        filtered_data = locust_data_with_geometry  # Use the collection with geometry

    # Get final count of features to process
    feature_count = filtered_data.size().getInfo()
    logging.info(
        f'Total features to process after all filters: {feature_count}')

    if feature_count == 0:
        logging.warning("No features remaining after filtering. Exiting.")
        return

    # Try to sort by 'Year' if it exists
    if 'Year' in filtered_data.first().propertyNames().getInfo():
        filtered_data = filtered_data.sort('Year', ascending=False)
        logging.info("Sorted features by 'Year' descending.")
    else:
        logging.warning("Property 'Year' not found. Skipping sort by Year.")

    # Create indexed collection using the server-side approach
    try:
        indexer = add_cumulative_index()
        indexed_collection = indexer(filtered_data)

        # Verify indexing worked correctly
        first_indexed_feature = indexed_collection.first().getInfo()
        if 'index' not in first_indexed_feature.get('properties', {}):
            logging.error(
                "Failed to add 'index' property to features. Check add_cumulative_index function.")
            return

        logging.info("Successfully created indexed collection.")
    except Exception as e:
        logging.error(f"Error creating indexed collection: {e}")
        return

    # Ensure start_index is valid
    if args.start_index >= feature_count:
        logging.error(
            f"Start index {args.start_index} is out of bounds (total features: {feature_count}). Exiting.")
        return

    # Load progress if available
    processed_indices, completed_count, failed_count, skipped_count = load_progress(
        args.progress_file)

    # Test mode - process a single point
    if args.test:
        process_in_test_mode(indexed_collection, args)
    else:
        # Batch mode - process multiple points
        process_in_batch_mode(
            indexed_collection,
            args,
            processed_indices,
            completed_count,
            failed_count,
            skipped_count
        )


if __name__ == "__main__":
    main()
