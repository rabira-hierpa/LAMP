"""
Geographic utilities for the locust processing package.
"""

import ee
import logging
import math
from typing import Dict, List, Optional, Union, Tuple

from ..config import BOUNDARIES_DATASET


def get_ethiopia_boundary() -> ee.FeatureCollection:
    """
    Get Ethiopia boundary as a FeatureCollection.

    Returns:
        ee.FeatureCollection: Feature collection containing Ethiopia boundary
    """
    return ee.FeatureCollection(BOUNDARIES_DATASET).filter(ee.Filter.eq('country_na', 'Ethiopia'))


def check_image_for_missing_data(image: ee.Image, geometry: ee.Geometry) -> bool:
    """
    Check if an image has any missing data in the specified bands.

    Args:
        image: Earth Engine image to check
        geometry: Region to check for missing data

    Returns:
        bool: True if missing data is detected, False otherwise
    """
    try:
        # Get a sample value for each band to check if data exists
        sample = image.sample(
            region=geometry,
            scale=1000,
            numPixels=1,
            seed=42,
            dropNulls=False
        ).first()

        # Convert to dictionary to check for nulls
        sample_dict = sample.toDictionary().getInfo()

        # Check if any values are None or NaN
        for key, value in sample_dict.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                logging.warning(f"Band {key} has null or NaN values")
                return True

        return False
    except Exception as e:
        logging.error(f"Error checking for missing data: {str(e)}")
        return True  # Assume data is missing if there's an error


def create_grid_around_point(point: ee.Feature,
                             grid_size: int = 7,
                             spatial_resolution: int = 1000) -> ee.FeatureCollection:
    """
    Create a grid of cells centered on a point.

    Args:
        point: Center point for the grid
        grid_size: Size of the grid (must be odd number)
        spatial_resolution: Spatial resolution in meters

    Returns:
        ee.FeatureCollection: Collection of grid cells as features
    """
    geometry = point.geometry()
    center_lat = geometry.coordinates().get(1)
    center_lon = geometry.coordinates().get(0)

    # Create a grid centered on the current location
    half_grid_size = math.floor(grid_size / 2)
    grid_cells = []

    # Calculate the step size in degrees based on the spatial resolution
    # Convert meters to approximate degrees
    step_size = spatial_resolution / 111000

    # Create a feature collection of grid cells
    for i in range(-half_grid_size, half_grid_size + 1):
        for j in range(-half_grid_size, half_grid_size + 1):
            cell_lon = ee.Number(center_lon).add(
                ee.Number(j).multiply(step_size))
            cell_lat = ee.Number(center_lat).add(
                ee.Number(i).multiply(step_size))

            cell = ee.Feature(
                ee.Geometry.Rectangle([
                    cell_lon.subtract(step_size/2),
                    cell_lat.subtract(step_size/2),
                    cell_lon.add(step_size/2),
                    cell_lat.add(step_size/2)
                ]),
                {
                    'row': i + half_grid_size,
                    'col': j + half_grid_size,
                    'center_lon': cell_lon,
                    'center_lat': cell_lat
                }
            )

            grid_cells.append(cell)

    return ee.FeatureCollection(grid_cells)


def date_to_ee_date(date_str: str) -> ee.Date:
    """
    Convert a date string to an Earth Engine date object.

    Args:
        date_str: Date string in format 'YYYY-MM-DD'

    Returns:
        ee.Date: Earth Engine date object
    """
    return ee.Date(date_str)


def parse_observation_date(feature: ee.Feature, feature_index: int) -> Optional[Tuple[ee.Date, str]]:
    """
    Parse the observation date from a feature.

    Args:
        feature: Feature containing 'Obs Date' property
        feature_index: Index for logging purposes

    Returns:
        Tuple of (ee.Date, formatted_date_str) if successful, None otherwise
    """
    try:
        # Server-side check if 'Obs Date' exists and is not null
        has_obs_date = feature.propertyNames().contains('Obs Date')
        obs_date_is_null = ee.Algorithms.IsEqual(feature.get('Obs Date'), None)

        # Use ee.Algorithms.If for conditional check server-side
        should_process_date = ee.Algorithms.If(
            has_obs_date.And(obs_date_is_null.Not()),
            True,
            False
        )

        if not should_process_date.getInfo():
            logging.warning(
                f"Feature {feature_index} has missing or null 'Obs Date'. Skipping.")
            return None

        # Get the raw date value safely
        obs_date_raw = feature.get('Obs Date')
        obs_date_client = obs_date_raw.getInfo()  # GetInfo only if needed

        # Validate client-side date format
        if not isinstance(obs_date_client, str) or '/' not in obs_date_client:
            logging.warning(
                f"Feature {feature_index} has invalid date format: {obs_date_client}. Skipping.")
            return None

        # Process the date parts carefully
        date_parts = obs_date_client.split(' ')[0].split('/')
        if len(date_parts) < 3:
            logging.warning(
                f"Feature {feature_index} invalid date parts: {date_parts}. Skipping.")
            return None

        month, day, year = date_parts[0].zfill(
            2), date_parts[1].zfill(2), date_parts[2]

        # Validate year
        import datetime
        if len(year) != 4 or not (1900 <= int(year) <= datetime.datetime.now().year + 1):
            logging.warning(
                f"Feature {feature_index} has invalid year: {year}. Skipping.")
            return None

        formatted_date = f"{year}-{month}-{day}"
        ee_date = ee.Date(formatted_date)  # Create EE Date server-side
        logging.info(
            f"Feature {feature_index}: Successfully parsed date: {formatted_date}")

        return ee_date, formatted_date

    except Exception as e:
        logging.warning(
            f"Error parsing date for feature {feature_index}: {e}. Raw date: {obs_date_client if 'obs_date_client' in locals() else 'N/A'}. Skipping.")
        return None


def get_date_range(center_date: ee.Date, days_back: int) -> Dict[str, ee.Date]:
    """
    Get date ranges for time lagged analysis.

    Args:
        center_date: Center date for time series
        days_back: Number of days to look back for each period

    Returns:
        Dict mapping period names to their start dates
    """
    return {
        "30": center_date.advance(-30, "days"),
        "60": center_date.advance(-60, "days"),
        "90": center_date.advance(-90, "days")
    }
