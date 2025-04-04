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
from .processing.export import create_export_task, create_test_export_task, start_export_task
from .processing.extraction import get_missing_variables
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


def process_single_feature(filtered_data: ee.FeatureCollection,
                           idx: int,
                           use_index: bool,
                           processed_indices: Set[int],
                           task_manager: TaskManager) -> bool:
    """
    Process a single feature by index or position.

    Args:
        filtered_data: FeatureCollection to process
        idx: Index of the feature to process
        use_index: Whether to use index property or direct position
        processed_indices: Set of already processed indices
        task_manager: TaskManager instance for task handling

    Returns:
        bool: True if the feature was processed successfully, False otherwise
    """
    if idx in processed_indices:
        logging.info(f"Skipping already processed feature index {idx}")
        return True

    try:
        # Get the feature either by index or position
        if use_index:
            # Filter by index property
            feature_collection = filtered_data.filter(
                ee.Filter.eq('index', idx)).limit(1)
        else:
            # Use direct pagination - sort by a property then take the idx-th element
            # Earth Engine doesn't support direct skip/limit, so we need to sort and take
            feature_collection = filtered_data.sort(
                'system:index').limit(1, ee.Number(idx).format("d"))

        # Check if any features were found
        count = feature_collection.size().getInfo()
        if count == 0:
            logging.warning(f"Feature at position {idx} not found. Skipping.")
            return False

        # Get the feature
        feature_to_process = ee.Feature(feature_collection.first())

        task_tuple = create_export_task(idx, feature_to_process)
        # Adds None if creation failed, handled by add_task
        task_manager.add_task(task_tuple)
        # Add to processed only if submitted or skipped by create_export_task
        processed_indices.add(idx)
        return True
    except ee.EEException as e:
        logging.error(f"Earth Engine error processing feature {idx}: {e}")
        return False
    except Exception as e:
        logging.error(f"General error processing feature {idx}: {str(e)}")
        return False


def process_in_test_mode(filtered_data: ee.FeatureCollection,
                         args: argparse.Namespace) -> None:
    """
    Process a single feature in test mode.

    Args:
        filtered_data: FeatureCollection to process
        args: Command-line arguments
    """
    logging.info(f"--- Running in Test Mode (Index: {args.start_index}) ---")
    try:
        single_point_collection = filtered_data.limit(1)

        # Check if any features were found
        single_point_count = single_point_collection.size()
        if single_point_count == 0:
            logging.error(
                f"Test mode: No feature found at position {args.start_index}. The collection may be smaller than expected.")
            return

        # Get the first (and only) feature
        single_point = ee.Feature(single_point_collection.first())
        logging.info(
            f"Test mode: Processing feature at position {args.start_index}")

        logging.info(f"Creating test export task...")
        task_tuple = create_test_export_task(args.start_index, single_point)

        if task_tuple:
            task, description = task_tuple
            logging.info(f"Starting test export task...")
            start_export_task(task, description)
            logging.info(f'Test export task started: {description}')

            # Wait for the task to complete or fail
            logging.info(f"Monitoring task status...")
            while True:
                status = task.status()
                state = status['state']
                logging.info(f"Test task status: {state} for {description}")

                if state not in ('READY', 'RUNNING'):
                    break

                time.sleep(10)

            final_status = task.status()
            logging.info(
                f"Test task '{description}' finished with state: {final_status['state']}")

            if final_status['state'] == 'COMPLETED':
                logging.info("Test export successful.")
            else:
                logging.error(
                    f"Test export failed: {final_status.get('error_message', 'Unknown error')}")

            # Log missing variables (if any)
            missing_vars = get_missing_variables()
            if missing_vars:
                logging.info(
                    f"Non-critical missing variables: {', '.join(missing_vars)}")
        else:
            logging.warning(
                f"Test export task for index {args.start_index} could not be created (check logs for reasons like missing data or invalid properties).")

    except ee.EEException as e:
        logging.error(f"Earth Engine error during test mode: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during test mode: {e}")
    finally:
        logging.info("--- Exiting Test Mode ---")


def process_in_batch_mode(filtered_data: ee.FeatureCollection,
                          args: argparse.Namespace,
                          processed_indices: Set[int],
                          initial_completed: int,
                          initial_failed: int,
                          initial_skipped: int) -> None:
    """
    Process features in batch mode.

    Args:
        filtered_data: FeatureCollection to process
        args: Command-line arguments
        processed_indices: Set of already processed indices
        initial_completed: Initial count of completed tasks
        initial_failed: Initial count of failed tasks
        initial_skipped: Initial count of skipped tasks
    """
    logging.info(f"--- Running in Batch Mode ---")

    # Check if collection has index property
    has_index = False
    try:
        # Check if collection has features first
        collection_size = filtered_data.size().getInfo()
        if collection_size > 0:
            # Safely get first feature
            first_feature = filtered_data.limit(1).first().getInfo()
            if first_feature and 'properties' in first_feature and 'index' in first_feature['properties']:
                has_index = True
                logging.info(
                    "Collection has 'index' property, using it for filtering.")
        else:
            logging.warning(
                "Collection is empty - cannot check for index property")
    except Exception as e:
        logging.info(f"Error checking for 'index' property: {str(e)}")

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
    feature_count = filtered_data.size().getInfo()
    max_idx = args.start_index + args.max_features if args.max_features else feature_count
    # Ensure we don't exceed the available features
    max_idx = min(max_idx, feature_count)

    task_manager.total_count = max_idx - args.start_index

    try:
        # Process in batches to avoid memory issues
        logging.info(
            f"Starting batch processing from index {args.start_index} to {max_idx-1}")

        # For balanced sampling between presence and absence (optional)
        if args.balanced_sampling and not (args.presence_only or args.absence_only):
            logging.info(
                "Using balanced sampling between presence and absence points")
            # This requires separately filtering presence and absence points
            presence_data = filtered_data.filter(
                ee.Filter.eq('Locust Presence', 'PRESENT'))
            absence_data = filtered_data.filter(
                ee.Filter.eq('Locust Presence', 'ABSENT'))

            presence_count = presence_data.size().getInfo()
            absence_count = absence_data.size().getInfo()

            logging.info(
                f"Found {presence_count} presence points and {absence_count} absence points")

            # Process equal numbers from each category
            max_balanced_count = min(presence_count, absence_count)
            processed_in_run = 0

            for i in range(max_balanced_count):
                if i < args.start_index:
                    continue

                if args.max_features and processed_in_run >= args.max_features:
                    break

                # Process a presence point
                presence_success = process_single_feature(
                    presence_data, i, False, processed_indices, task_manager)
                if presence_success:
                    processed_in_run += 1

                if args.max_features and processed_in_run >= args.max_features:
                    break

                # Process an absence point
                absence_success = process_single_feature(
                    absence_data, i, False, processed_indices, task_manager)
                if absence_success:
                    processed_in_run += 1

                # Save progress periodically
                if processed_in_run > 0 and processed_in_run % 10 == 0:
                    save_progress(
                        args.progress_file,
                        list(processed_indices),
                        task_manager.completed_count,
                        task_manager.failed_count,
                        task_manager.skipped_count
                    )

            # Process any remaining points if requested
            if not args.balance_only:
                remaining_to_process = args.max_features - \
                    processed_in_run if args.max_features else float('inf')

                if remaining_to_process > 0 and presence_count > max_balanced_count:
                    remaining_presence = min(
                        presence_count - max_balanced_count, remaining_to_process)
                    logging.info(
                        f"Processing {remaining_presence} additional presence points")

                    for i in range(max_balanced_count, max_balanced_count + remaining_presence):
                        success = process_single_feature(
                            presence_data, i, False, processed_indices, task_manager)
                        if success:
                            processed_in_run += 1

                        remaining_to_process -= 1
                        if remaining_to_process <= 0:
                            break

                        # Save progress periodically
                        if processed_in_run % 10 == 0:
                            save_progress(
                                args.progress_file,
                                list(processed_indices),
                                task_manager.completed_count,
                                task_manager.failed_count,
                                task_manager.skipped_count
                            )

                if remaining_to_process > 0 and absence_count > max_balanced_count:
                    remaining_absence = min(
                        absence_count - max_balanced_count, remaining_to_process)
                    logging.info(
                        f"Processing {remaining_absence} additional absence points")

                    for i in range(max_balanced_count, max_balanced_count + remaining_absence):
                        success = process_single_feature(
                            absence_data, i, False, processed_indices, task_manager)
                        if success:
                            processed_in_run += 1

                        # Save progress periodically
                        if processed_in_run % 10 == 0:
                            save_progress(
                                args.progress_file,
                                list(processed_indices),
                                task_manager.completed_count,
                                task_manager.failed_count,
                                task_manager.skipped_count
                            )

        else:
            # Process sequentially (original method)
            processed_in_run = 0
            for current_index in range(args.start_index, max_idx):
                success = process_single_feature(
                    filtered_data, current_index, has_index, processed_indices, task_manager)
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


def debug_collection(collection, description="Collection", limit=3):
    """
    Debug helper function to inspect a collection.

    Args:
        collection: Earth Engine FeatureCollection to inspect
        description: Description of the collection for logging
        limit: Number of features to inspect

    Returns:
        None
    """
    try:
        logging.info(f"\n--- DEBUG: {description} ---")

        # Get collection size
        size = collection.size().getInfo()
        logging.info(f"Collection size: {size}")

        if size == 0:
            logging.warning("Collection is empty!")
            return

        # Try to safely get first feature
        try:
            first_feature_info = collection.limit(1).first().getInfo()
            if first_feature_info and 'properties' in first_feature_info:
                prop_keys = list(first_feature_info['properties'].keys())
                logging.info(f"First feature property keys: {prop_keys}")

                # Check key properties
                for key in ['Obs Date', 'Locust Presence', 'Year', 'Longitude', 'Latitude']:
                    if key in first_feature_info['properties']:
                        value = first_feature_info['properties'][key]
                        logging.info(f"  {key}: {value}")
                    else:
                        logging.info(f"  {key}: Not found")
            else:
                logging.warning("Could not retrieve feature properties")
        except Exception as e:
            logging.error(f"Error inspecting first feature: {str(e)}")

        # Test pagination
        try:
            if size > 1:
                # Use system:index to sort and format the number as a string for the second parameter
                second_feature = collection.sort('system:index').limit(
                    1, ee.Number(1).format("d")).first().getInfo()
                if second_feature:
                    logging.info(f"Second feature found with pagination")
        except Exception as e:
            logging.error(f"Error testing pagination: {str(e)}")

    except Exception as e:
        logging.error(f"Error debugging collection: {str(e)}")

    logging.info("--- END DEBUG ---\n")


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
    parser.add_argument('--min-year', type=int, default=None,
                        help='Minimum year to include (e.g., 2015)')
    parser.add_argument('--balanced-sampling', action='store_true',
                        help='Use balanced sampling between presence and absence points')
    parser.add_argument('--balance-only', action='store_true',
                        help='Process only balanced samples (equal number of presence and absence)')
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

    original_count = locust_data.size().getInfo()

    if original_count > 0:
        try:
            first_feature = locust_data.limit(1).first()
            property_names = first_feature.propertyNames()
            logging.info(
                f"First feature properties: {property_names.getInfo()}")
        except Exception as e:
            logging.warning(
                f"Could not get first feature properties: {str(e)}")
    else:
        logging.warning(
            "Collection is empty - cannot check feature properties")

    # Filter by minimum year if specified
    if args.min_year is not None:
        locust_data = locust_data.filter(
            ee.Filter.gte('Year', args.min_year))
        count_after_year_filter = locust_data.size().getInfo()
        logging.info(
            f"Features after filtering for year >= {args.min_year}: {count_after_year_filter} (removed {count_after_geom_filter - count_after_year_filter})")
        count_after_geom_filter = count_after_year_filter

    # Filter for presence and absence points based on the geometry-added collection
    if args.presence_only:
        logging.info("Processing only presence points")
        filtered_data = locust_data.filter(
            ee.Filter.eq('Locust Presence', 'PRESENT'))
    elif args.absence_only:
        logging.info("Processing only absence points")
        filtered_data = locust_data.filter(
            ee.Filter.eq('Locust Presence', 'ABSENT'))
    else:
        filtered_data = locust_data  # Use the collection with geometry

    # Get final count of features to process
    feature_count = filtered_data.size().getInfo()
    logging.info(
        f'Total features to process after all filters: {feature_count}')

    if feature_count == 0:
        logging.warning("No features remaining after filtering. Exiting.")
        return

    # Ensure start_index is valid
    if args.start_index >= feature_count:
        logging.error(
            f"Start index {args.start_index} is out of bounds (total features: {feature_count}). Exiting.")
        return

    # Load progress if available
    processed_indices, completed_count, failed_count, skipped_count = load_progress(
        args.progress_file)

    # Log processing counts for presence/absence
    try:
        presence_count = filtered_data.filter(ee.Filter.eq(
            'Locust Presence', 'PRESENT')).size().getInfo()
        absence_count = filtered_data.filter(ee.Filter.eq(
            'Locust Presence', 'ABSENT')).size().getInfo()
        logging.info(
            f"Processing {presence_count} presence points and {absence_count} absence points")
    except Exception as e:
        logging.warning(f"Could not count presence/absence points: {e}")

    # Test mode - process a single point
    if args.test:
        logging.info("Running in test mode with detailed debugging...")
        # Debug the filtered collection right before processing
        # debug_collection(filtered_data, "Final Filtered Collection")
        process_in_test_mode(filtered_data, args)
    else:
        # Batch mode - process multiple points
        process_in_batch_mode(
            filtered_data,
            args,
            processed_indices,
            completed_count,
            failed_count,
            skipped_count
        )


if __name__ == "__main__":
    main()
