"""
Export functionality for the locust processing package.
"""

import ee
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

from ..utils.geo_utils import parse_observation_date
from ..processing.extraction import extract_time_lagged_data, verify_presence_value
from ..config import COMMON_SCALE, COMMON_PROJECTION, EXPORT_FOLDER, POINT_BUFFER_METERS, MAX_PIXELS


def create_export_task(feature_index: int, feature: ee.Feature) -> Optional[Tuple[ee.batch.Task, str]]:
    """
    Create an export task for a feature.

    Args:
        feature_index: Index of the feature for logging
        feature: Earth Engine feature to process

    Returns:
        Tuple of (export_task, description) if successful, None otherwise
    """
    try:
        # Parse observation date
        date_result = parse_observation_date(feature, feature_index)
        if date_result is None:
            return None

        ee_date, formatted_date = date_result

        # Verify locust presence value
        presence = verify_presence_value(feature, feature_index)
        if presence is None:
            return None

        # Set the parsed date on the feature
        feature_with_parsed_date = feature.set('parsed_date', ee_date)

        # Extract time-lagged data
        time_lagged_data = extract_time_lagged_data(feature_with_parsed_date)

        # Skip if data is missing
        if time_lagged_data is None:
            logging.warning(
                f"Missing environmental data for feature {feature_index}. Skipping task creation.")
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

        # Create export task
        export_description = f'locust_{formatted_date}_label_{1 if presence == "PRESENT" else 0}_{feature_index + 1}'

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
