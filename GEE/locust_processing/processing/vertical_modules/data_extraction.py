"""
Data extraction module for the locust processing package.

This module handles the extraction of time-lagged environmental data for locust prediction.
"""

import ee
import datetime
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Import vertical-specific modules
from .vegetation_indices import compute_ndvi, compute_evi
from .temperature_processing import compute_lst
from .moisture_analysis import compute_precipitation, compute_soil_moisture, compute_ndwi
from .wind_processing import compute_wind_components
from .index_calculation import calculate_all_indices, has_critical_data_missing

from ...utils.geo_utils import check_image_for_missing_data, create_grid_around_point, get_date_range
from ...config import SPATIAL_RESOLUTION, GRID_SIZE, FAO_REPORT_ASSET_ID, MAX_PIXELS


# Global tracker for missing data
missing_data_variables: List[str] = []


def get_missing_variables() -> List[str]:
    """
    Get the list of missing variables for the current extraction.

    Returns:
        List[str]: List of missing variable names
    """
    global missing_data_variables
    return missing_data_variables


def compute_variable(collection_id: str, bands: List[str], reducer: str, lag: str,
                    geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> Optional[ee.Image]:
    """
    Compute a single variable from an image collection with error handling and fallbacks.

    Args:
        collection_id: Earth Engine image collection ID
        bands: List of bands to select
        reducer: Type of reducer to apply ("mean" or "sum")
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        ee.Image: Computed variable or None if data is missing
    """
    global missing_data_variables

    try:
        reducer_fn = ee.Reducer.mean() if reducer == "mean" else ee.Reducer.sum()

        # Filter collection by bounds and date
        collection = ee.ImageCollection(collection_id) \
            .filterBounds(geometry) \
            .filterDate(date_range[lag], date_range["0"])

        # Check if collection is empty
        collection_size = collection.size().getInfo()
        if collection_size == 0:
            variable_name = f"{collection_id} {bands[0]} {lag}"
            logging.warning(f"Empty collection for {variable_name}")
            missing_data_variables.append(variable_name)
            return ee.Image(0).rename(f"{bands[0]}_{lag}")

        # Process the collection
        return collection.select(bands) \
            .reduce(reducer_fn) \
            .rename(f"{bands[0]}_{lag}")

    except Exception as e:
        variable_name = f"{collection_id} {bands[0]} {lag}"
        logging.error(f"Error computing {variable_name}: {str(e)}")
        missing_data_variables.append(variable_name)
        return ee.Image(0).rename(f"{bands[0]}_{lag}")
