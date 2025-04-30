"""
Export functionality for the locust processing package.
"""

import ee
import logging
import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

from ..utils.geo_utils import parse_observation_date
from ..processing.extraction import extract_time_lagged_data, verify_presence_value, get_missing_variables, has_critical_data_missing
from ..config import COMMON_SCALE, COMMON_PROJECTION, EXPORT_FOLDER, POINT_BUFFER_METERS, MAX_PIXELS


def create_export_task(feature_index: int, feature: ee.Feature, country: str = None, dry_run: bool = False) -> Optional[Tuple[ee.batch.Task, str]]:
    """
    Create an export task for a feature.

    Args:
        feature_index: Index of the feature for logging
        feature: Earth Engine feature to process
        country: Optional country name to use in export folder
        dry_run: If True, simulate task creation without actually creating it

    Returns:
        Tuple of (export_task, description) if successful, None otherwise
    """
    try:
        # Parse observation date
        date_result = parse_observation_date(feature, feature_index)
        if date_result is None:
            # Skip feature
            logging.warning(
                f"No valid observation date found for feature {feature_index}. Skipping task creation.")
            return None

        else:
            ee_date, formatted_date = date_result

        # Verify locust presence value
        presence = verify_presence_value(feature, feature_index)
        if presence is None:
            logging.warning(
                f"Invalid presence value for feature {feature_index}. Skipping task creation.")
            return None

        # Set the parsed date on the feature
        feature_with_parsed_date = feature.set('parsed_date', ee_date)

        # Extract time-lagged data
        time_lagged_data = extract_time_lagged_data(feature_with_parsed_date)

        # Skip if critical data is missing
        if time_lagged_data is None:
            missing_vars = get_missing_variables()
            if missing_vars:
                logging.warning(
                    f"Missing variables for feature {feature_index}: {', '.join(missing_vars)}")
            logging.warning(
                f"Skipping export task creation for feature {feature_index} due to missing data")
            return None

        # Prepare for export
        time_lagged_data = time_lagged_data.toFloat()

        # Ensure geometry exists before buffering
        feature_geometry = feature.geometry()
        if feature_geometry is None:
            logging.warning(
                f"Feature {feature_index} has no geometry. Skipping.")
            return None

        patch_geometry = feature_geometry.buffer(
            POINT_BUFFER_METERS)  # Buffer by configured amount

        # Create multi-band image with label
        multi_band_image = ee.Image.cat([
            time_lagged_data,
            ee.Image.constant(
                1 if presence == 'PRESENT' else 0).toFloat().rename('label')
        ]).clip(patch_geometry)

        # Create export task with a detailed name to facilitate tracking
        presence_value = 1 if presence == 'PRESENT' else 0

        # Set consistent description name including country indicator if provided
        if country:
            export_description = f'locust_{formatted_date}_label_{presence_value}_{country}_idx_{feature_index}'
        else:
            export_description = f'locust_{formatted_date}_label_{presence_value}_idx_{feature_index}'

        # Log non-critical missing variables (if any)
        missing_vars = get_missing_variables()
        if missing_vars:
            logging.info(
                f"Non-critical missing variables for feature {feature_index}: {', '.join(missing_vars)}")

        if dry_run:
            logging.info(
                f"[DRY RUN] Would create export task for feature {feature_index}:")
            logging.info(f"  Description: {export_description}")
            logging.info(f"  Folder: {EXPORT_FOLDER}")
            logging.info(f"  Scale: {COMMON_SCALE}")
            logging.info(f"  CRS: {COMMON_PROJECTION}")
            logging.info(f"  Max Pixels: {MAX_PIXELS}")
            return None, export_description

        export_task = ee.batch.Export.image.toDrive(
            image=multi_band_image,
            description=export_description,
            scale=COMMON_SCALE,
            region=patch_geometry,
            maxPixels=MAX_PIXELS,
            crs=COMMON_PROJECTION,
            folder=EXPORT_FOLDER
        )
        logging.info(
            f"Created export task for feature {feature_index}: {export_description}")
        return export_task, export_description

    except ee.EEException as e:
        logging.error(
            f"Earth Engine error creating export task for feature {feature_index}: {str(e)}. Skipping.")
        return None
    except Exception as e:
        logging.error(
            f"General error creating export task for feature {feature_index}: {str(e)}. Skipping.")
        return None


def create_test_export_task(feature_index: int, feature: ee.Feature) -> Optional[Tuple[ee.batch.Task, str]]:
    """
    Create an export task with additional logging for testing purposes.

    Args:
        feature_index: Index of the feature for logging
        feature: Earth Engine feature to process

    Returns:
        Tuple of (export_task, description) if successful, None otherwise
    """
    logging.info(f"Creating test export task for feature {feature_index}")

    # Log feature properties for debugging
    try:
        properties = feature.propertyNames().getInfo()
        logging.info(f"Feature properties: {properties}")
    except Exception as e:
        logging.warning(f"Couldn't get feature properties: {e}")

    # Create and return the export task
    return create_export_task(feature_index, feature)


def start_export_task(task: ee.batch.Task, description: str) -> bool:
    """
    Start an export task and return success/failure.

    Args:
        task: Earth Engine export task
        description: Task description for logging

    Returns:
        bool: True if task started successfully, False otherwise
    """
    try:
        task.start()
        logging.info(f"Started export task: {description}")
        return True
    except Exception as e:
        logging.error(f"Failed to start export task '{description}': {str(e)}")
        return False


def get_task_status(task: ee.batch.Task) -> Dict[str, Any]:
    """
    Get the status of a task.

    Args:
        task: Earth Engine task

    Returns:
        Dict: Status dictionary with 'state' and possibly 'error_message'
    """
    try:
        return task.status()
    except Exception as e:
        logging.error(f"Failed to get task status: {str(e)}")
        return {'state': 'ERROR', 'error_message': str(e)}
