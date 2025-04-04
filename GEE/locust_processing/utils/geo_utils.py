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
    Parse the observation date from a feature with a pre-formatted date.

    Args:
        feature: Feature containing 'formatted_date' property
        feature_index: Index for logging purposes

    Returns:
        Tuple of (ee.Date, formatted_date_str) if successful, None otherwise
    """
    try:
        # Get the formatted date (already in YYYY-MM-DD format)
        formatted_date = feature.get('formatted_date')

        # Create EE Date object server-side
        ee_date = ee.Date(formatted_date)

        return ee_date, formatted_date.getInfo()

    except Exception as e:
        logging.warning(
            f"Error parsing date for feature {feature_index}: {e}. Skipping.")
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
