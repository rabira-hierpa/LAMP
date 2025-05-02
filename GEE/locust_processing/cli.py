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
                           object_id: int,
                           processed_object_ids: Set[int],
                           task_manager: TaskManager,
                           total_to_process: int,
                           country: str = None,
                           progress_file: str = None,
                           dry_run: bool = False,
                           ) -> bool:
    """
    Process a single feature by OBJECTID.

    Args:
        filtered_data: FeatureCollection to process
        object_id: OBJECTID of the feature to process
        processed_object_ids: Set of already processed OBJECTIDs
        task_manager: TaskManager instance for task handling
        total_to_process: Total number of features to process
        country: Optional country name to use in export folder
        progress_file: Optional path to the progress file
        dry_run: If True, simulate task creation without actually creating it

    Returns:
        bool: True if the feature was queued successfully, False otherwise
    """
    if object_id in processed_object_ids:
        logging.info(
            f"⏭️ Skipping already processed feature with OBJECTID {object_id}")
        return True

    try:
        # Filter by OBJECTID from the filtered_data
        feature_collection = filtered_data.filter(
            ee.Filter.eq('OBJECTID', object_id))

        # Check if any features were found
        count = feature_collection.size().getInfo()
        if count == 0:
            logging.warning(
                f"🚫 Feature with OBJECTID {object_id} not found. Skipping.")
            task_manager.increment_skipped(object_id)
            return False

        # Get the feature
        feature_to_process = ee.Feature(feature_collection.first())
        # Create the export task
        try:
            task_tuple = create_export_task(
                object_id, feature_to_process, country, dry_run)
            if task_tuple is None:
                logging.error(
                    f"🚫 Could not create export task for feature with OBJECTID {object_id}")
                task_manager.increment_skipped(object_id)
                return False

            # Add to task manager
            task, description = task_tuple
            if not dry_run:
                task_manager.add_task(
                    task, description, total_to_process, object_id)
            return True
        except Exception as e:
            logging.error(
                f"Error creating/adding task for feature with OBJECTID {object_id}: {str(e)}")
            task_manager.increment_skipped(object_id)
            return False

    except ee.EEException as e:
        logging.error(
            f"Earth Engine error processing feature with OBJECTID {object_id}: {e}")
        task_manager.increment_skipped(object_id)
        return False
    except Exception as e:
        logging.error(
            f"General error processing feature with OBJECTID {object_id}: {str(e)}")
        task_manager.increment_skipped(object_id)
        return False


def process_features_parallel(filtered_data: ee.FeatureCollection,
                              object_ids: List[int],
                              processed_object_ids: Set[int],
                              task_manager: TaskManager,
                              batch_size: int = 4,
                              country: str = None,
                              progress_file: str = None,
                              dry_run: bool = False,
                              max_features: int = None) -> None:
    """
    Process features in parallel batches.

    Args:
        filtered_data: FeatureCollection to process
        object_ids: List of OBJECTIDs to process
        processed_object_ids: Set of already processed OBJECTIDs
        task_manager: TaskManager instance for task handling
        batch_size: Number of features to process in each batch
        country: Optional country name to use in export folder
        progress_file: Optional path to the progress file
        dry_run: If True, simulate task creation without actually creating it
        max_features: Maximum number of features to process
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    # Filter out already processed OBJECTIDs
    object_ids_to_process = [
        oid for oid in object_ids if oid not in processed_object_ids]

    # Apply max_features limit if specified
    if max_features is not None and len(object_ids_to_process) > max_features:
        logging.info(
            f"⬇︎ Limiting to {max_features} features as requested")
        object_ids_to_process = object_ids_to_process[:max_features]

    total_to_process = len(object_ids_to_process)

    # Update the task manager's total_to_process value
    task_manager.total_to_process = total_to_process

    if total_to_process == 0:
        logging.info("No new features to process")
        return

    logging.info(
        f"🔄 Processing {total_to_process} features in parallel batches of {batch_size}")

    # Process in batches to avoid memory issues
    for batch_start in range(0, total_to_process, batch_size):
        batch_end = min(batch_start + batch_size, total_to_process)
        batch_object_ids = object_ids_to_process[batch_start:batch_end]

        logging.info(
            f"🚧Processing batch {batch_start//batch_size + 1} with {len(batch_object_ids)} features")

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
            # Submit all tasks in the batch
            future_to_oid = {
                executor.submit(process_single_feature, filtered_data, oid, processed_object_ids, task_manager, total_to_process, country, progress_file, dry_run): oid
                for oid in batch_object_ids
            }

            # Process results as they complete
            for future in as_completed(future_to_oid):
                oid = future_to_oid[future]
                try:
                    success = future.result()
                    if success:
                        logging.info(
                            f"✅ Successfully queued feature with OBJECTID {oid}")
                    else:
                        logging.warning(
                            f"🚫 Failed to queue feature with OBJECTID {oid}")
                except Exception as e:
                    logging.error(
                        f"💥 Error processing feature with OBJECTID {oid}: {str(e)}")

        # Small pause between batches to avoid overwhelming Earth Engine
        time.sleep(0.5)


def process_in_test_mode(filtered_data: ee.FeatureCollection,
                         args: argparse.Namespace) -> None:
    """
    Process a single feature in test mode.

    Args:
        filtered_data: FeatureCollection to process
        args: Command-line arguments
    """
    logging.info(f"--- 🧪 Running in Test Mode (Index: {args.start_index}) ---")
    try:
        single_point_collection = filtered_data.limit(1)

        # Check if any features were found
        single_point_count = single_point_collection.size()
        if single_point_count == 0:
            logging.error(
                f"❌ Test mode: No feature found at position {args.start_index}. The collection may be smaller than expected.")
            return

        # Get the first (and only) feature
        single_point = ee.Feature(single_point_collection.first())
        logging.info(
            f"🔍 Test mode: Processing feature at position {args.start_index}")

        logging.info(f"🔧 Creating test export task...")
        task_tuple = create_test_export_task(args.start_index, single_point)

        if task_tuple:
            task, description = task_tuple
            logging.info(f"▶️ Starting test export task...")
            start_export_task(task, description)
            logging.info(f'🚀 Test export task started: {description}')

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
                          processed_object_ids: Set[int],
                          initial_completed: int,
                          initial_failed: int,
                          initial_skipped: int) -> None:
    """
    Process features in batch mode.

    Args:
        filtered_data: FeatureCollection to process
        args: Command-line arguments
        processed_object_ids: Set of already processed OBJECTIDs
        initial_completed: Initial count of completed tasks
        initial_failed: Initial count of failed tasks
        initial_skipped: Initial count of skipped tasks
    """
    logging.info(f"--- 🚀 Running in Batch Mode ---")

    # Initialize task manager for batch processing
    task_manager = TaskManager(
        max_concurrent=MAX_CONCURRENT_TASKS,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY_SECONDS,
        progress_file=args.progress_file
    )

    # Pass initial loaded counts to task manager
    task_manager.completed_count = initial_completed
    task_manager.failed_count = initial_failed
    task_manager.skipped_count = initial_skipped

    # Initialize processed object IDs
    for obj_id in processed_object_ids:
        task_manager.processed_object_ids.add(obj_id)

    logging.info(
        f"Loaded {len(task_manager.processed_object_ids)} processed object IDs from progress file")

    try:
        # Get all OBJECTIDs from the collection
        object_ids = filtered_data.aggregate_array('OBJECTID').getInfo()
        if not object_ids:
            logging.error("No OBJECTIDs found in the collection")
            return

        # Calculate total based on the range we intend to process
        total_object_ids = len(object_ids)
        if args.max_features:
            logging.info(
                f"User requested to limit processing to {args.max_features} features")
            total_object_ids = min(total_object_ids, args.max_features)

        # Set both total_count and total_to_process to ensure consistent reporting
        task_manager.total_count = total_object_ids
        task_manager.total_to_process = total_object_ids

        # Process in batches to avoid memory issues
        logging.info(
            f"Starting batch processing with {total_object_ids} features")

        # For balanced sampling between presence and absence (optional)
        if args.balanced_sampling and not (args.presence_only or args.absence_only):
            logging.info(
                "⚖️ Using balanced sampling between presence and absence points")

            # Filter presence and absence points
            presence_data = filtered_data.filter(
                ee.Filter.eq('Locust Presence', 'PRESENT'))
            absence_data = filtered_data.filter(
                ee.Filter.eq('Locust Presence', 'ABSENT'))

            # Get OBJECTIDs for presence and absence
            presence_object_ids = presence_data.aggregate_array(
                'OBJECTID').getInfo()
            absence_object_ids = absence_data.aggregate_array(
                'OBJECTID').getInfo()

            logging.info(
                f"📊 Found {len(presence_object_ids)} presence points and {len(absence_object_ids)} absence points")

            # Process equal numbers from each category
            max_balanced_count = min(
                len(presence_object_ids), len(absence_object_ids))
            if args.max_features:
                # Half to each category
                max_balanced_count = min(
                    max_balanced_count, args.max_features // 2)

            logging.info(
                f"Processing {max_balanced_count} presence points and {max_balanced_count} absence points")
            # Process presence points in parallel
            if presence_object_ids:
                presence_object_ids = presence_object_ids[:max_balanced_count]
                logging.info(
                    f"Processing {len(presence_object_ids)} presence points in parallel")
                process_features_parallel(
                    presence_data, presence_object_ids, task_manager.processed_object_ids, task_manager, args.batch_size, args.country, args.progress_file, args.dry_run)

            # Process absence points in parallel
            if absence_object_ids:
                absence_object_ids = absence_object_ids[:max_balanced_count]
                logging.info(
                    f"Processing {len(absence_object_ids)} absence points in parallel")
                process_features_parallel(
                    absence_data, absence_object_ids, task_manager.processed_object_ids, task_manager, args.batch_size, args.country, args.progress_file, args.dry_run)

        else:
            # Process all features in parallel
            process_features_parallel(
                filtered_data, object_ids, task_manager.processed_object_ids, task_manager,
                args.batch_size, args.country, args.progress_file, args.dry_run, args.max_features)

        # Wait for all tasks to complete
        logging.info(
            "⏳ All features submitted to queue. Waiting for tasks to complete...")
        final_completed, final_failed, final_skipped = task_manager.wait_until_complete()

        logging.info(
            f"🏁 Batch processing completed: ✅ {final_completed} successful, ❌ {final_failed} failed, ⏭️ {final_skipped} skipped")

    except KeyboardInterrupt:
        logging.info("⚠️ Interrupted by user. Saving progress...")
        # Tasks will be saved through TaskManager.shutdown()
    except Exception as e:
        logging.error(f"❌ Error in main batch processing loop: {str(e)}")
    finally:
        # Shutdown task manager
        logging.info("🛑 Shutting down task manager...")
        task_manager.shutdown()

        logging.info("✨ Script completed. Final progress saved.")


def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Export locust data from Earth Engine')
    parser.add_argument('--test', action='store_true',
                        help='Run with a single test point')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of features to process in one batch')
    parser.add_argument('--max-features', type=int, default=None,
                        help='Maximum number of features to process')
    parser.add_argument('--presence-only', action='store_true',
                        help='Process only presence points')
    parser.add_argument('--absence-only', action='store_true',
                        help='Process only absence points')
    parser.add_argument('--min-year', type=int, default=None,
                        help='Minimum year to include (e.g., 2015)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date in YYYY-MM-DD format (e.g., 2015-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date in YYYY-MM-DD format (e.g., 2020-12-31)')
    parser.add_argument('--balanced-sampling', action='store_true',
                        help='Use balanced sampling between presence and absence points')
    parser.add_argument('--balance-only', action='store_true',
                        help='Process only balanced samples (equal number of presence and absence)')
    parser.add_argument('--progress-file', type=str, default=DEFAULT_PROGRESS_FILE,
                        help='File to save/load progress')
    parser.add_argument('--log-file', type=str, default=DEFAULT_LOG_FILE,
                        help='Log file name')
    parser.add_argument('--country', type=str, default=None,
                        help='Filter data by country name (e.g., "Ethiopia")')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode (no actual exports will be created)')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_file)

    # Initialize Earth Engine
    initialize_ee()

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

    # Filter by date range if specified
    if args.start_date is not None or args.end_date is not None:
        try:
            if args.start_date:
                start_date = ee.Date(args.start_date)
                locust_data = locust_data.filter(
                    ee.Filter.gte('Obs Date', start_date))
            if args.end_date:
                end_date = ee.Date(args.end_date)
                locust_data = locust_data.filter(
                    ee.Filter.lte('Obs Date', end_date))

            count_after_date_filter = locust_data.size().getInfo()
            logging.info(
                f"Features after date filtering: {count_after_date_filter} (removed {original_count - count_after_date_filter})")
            original_count = count_after_date_filter
        except Exception as e:
            logging.error(f"Error applying date filter: {str(e)}")
            return

    # Filter by minimum year if specified
    if args.min_year is not None:
        locust_data = locust_data.filter(
            ee.Filter.gte('Year', args.min_year))
        count_after_year_filter = locust_data.size().getInfo()
        logging.info(
            f"Features after filtering for year >= {args.min_year}: {count_after_year_filter} (removed {original_count - count_after_year_filter})")
        original_count = count_after_year_filter

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

    # Filter by country if specified
    if args.country is not None:
        logging.info(f"Filtering data for country: {args.country}")
        filtered_data = filtered_data.filter(
            ee.Filter.eq('Country', args.country))
        logging.info(
            f"🗺️ Features after country filter: {filtered_data.size().getInfo()}")

    # Get final count of features to process
    feature_count = filtered_data.size().getInfo()
    logging.info(
        f'Total features to process after all filters: {feature_count}')

    if feature_count == 0:
        logging.warning("🚫 No features remaining after filtering. Exiting.")
        return

    # Load progress if available
    processed_object_ids, completed_count, failed_count, skipped_count = load_progress(
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

    # Handle dry run mode
    if args.dry_run:
        logging.info(
            "🔍 DRY RUN MODE ENABLED - No actual exports will be performed")
        if args.test:
            logging.info(
                f"🧪 Would run test mode on a single point")
        else:
            logging.info(
                f"📊 Would process {feature_count} features")
            if args.presence_only:
                logging.info(f"🟢 Would only process presence points")
            elif args.absence_only:
                logging.info(f"🔴 Would only process absence points")
            if args.balanced_sampling:
                logging.info(
                    f"⚖️ Would use balanced sampling between presence and absence points")
            if args.country:
                logging.info(f"🗺️ Would filter for country: {args.country}")
        return

    # Test mode - process a single point
    if args.test:
        logging.info("Running in test mode with detailed debugging...")
        process_in_test_mode(filtered_data, args)
    else:
        # Batch mode - process multiple points
        process_in_batch_mode(
            filtered_data,
            args,
            processed_object_ids,
            completed_count,
            failed_count,
            skipped_count
        )


if __name__ == "__main__":
    main()
