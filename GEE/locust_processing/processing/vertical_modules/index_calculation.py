"""
Index calculation module for the locust processing package.

This module integrates functionality from all vertical-specific modules
to calculate environmental indices for locust prediction.
"""

import ee
import logging
from typing import Dict, List, Optional, Union

# Import vertical-specific modules
from .vegetation_indices import calculate_vhi
from .temperature_processing import calculate_tci, calculate_tvdi
from .moisture_analysis import calculate_ndwi
from .wind_processing import calculate_wind_speed, calculate_wind_direction


def calculate_all_indices(image: ee.Image) -> ee.Image:
    """
    Calculate all vegetation and drought indices.

    Args:
        image: Earth Engine image with required bands

    Returns:
        Earth Engine image with all indices added
    """
    # Calculate temperature indices
    image = calculate_tci(image)
    
    # Calculate vegetation indices
    image = calculate_vhi(image)
    
    # Calculate moisture indices
    image = calculate_ndwi(image)
    
    # Calculate temperature-vegetation indices
    image = calculate_tvdi(image)
    
    # Calculate wind indices
    image = calculate_wind_speed(image)
    image = calculate_wind_direction(image)
    
    return image


def has_critical_data_missing(missing_variables: List[str]) -> bool:
    """
    Check if critical variables are missing from the extracted data.

    Args:
        missing_variables: List of missing variable names

    Returns:
        bool: True if critical data is missing, False otherwise
    """
    # Define critical variables that must be present
    critical_variables = [
        "MODIS/061/MOD13Q1 NDVI 30",
        "MODIS/061/MOD13Q1 NDVI 60",
        "MODIS/061/MOD13Q1 NDVI 90",
        "MODIS/061/MOD11A2 LST_Day_1km 30",
        "MODIS/061/MOD11A2 LST_Day_1km 60",
        "MODIS/061/MOD11A2 LST_Day_1km 90"
    ]

    # Check if any critical variable is missing
    for var in critical_variables:
        if var in missing_variables:
            logging.error(f"Critical variable missing: {var}")
            return True

    return False
