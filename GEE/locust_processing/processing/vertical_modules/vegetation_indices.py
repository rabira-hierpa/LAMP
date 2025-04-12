"""
Vegetation indices calculation module for the locust processing package.

This module handles NDVI/EVI related calculations and processing.
"""

import ee
import logging
from typing import Dict, List, Optional, Union

# Global reference to Ethiopia boundary - will be initialized in the main module
et_boundary = None


def set_region_boundary(boundary: ee.FeatureCollection) -> None:
    """
    Set the region boundary for index calculations.

    Args:
        boundary: Earth Engine feature collection defining the boundary
    """
    global et_boundary
    et_boundary = boundary


def compute_ndvi(collection_id: str, bands: List[str], reducer: str, lag: str,
                geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> Optional[ee.Image]:
    """
    Compute NDVI from an image collection with error handling and fallbacks.

    Args:
        collection_id: Earth Engine image collection ID
        bands: List of bands to select
        reducer: Type of reducer to apply ("mean" or "sum")
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        ee.Image: Computed NDVI or None if data is missing
    """
    try:
        reducer_fn = ee.Reducer.mean() if reducer == "mean" else ee.Reducer.sum()

        # Filter collection by bounds and date
        collection = ee.ImageCollection(collection_id) \
            .filterBounds(geometry) \
            .filterDate(date_range[lag], date_range["0"])

        # Check if collection is empty
        collection_size = collection.size().getInfo()
        if collection_size == 0:
            variable_name = f"{collection_id} NDVI {lag}"
            logging.warning(f"Empty collection for {variable_name}")
            return ee.Image(0).rename(f"NDVI_{lag}")

        # Process the collection
        return collection.select(bands) \
            .reduce(reducer_fn) \
            .rename(f"{bands[0]}_{lag}")

    except Exception as e:
        variable_name = f"{collection_id} NDVI {lag}"
        logging.error(f"Error computing {variable_name}: {str(e)}")
        return ee.Image(0).rename(f"NDVI_{lag}")


def compute_evi(collection_id: str, bands: List[str], reducer: str, lag: str,
               geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> Optional[ee.Image]:
    """
    Compute EVI from an image collection with error handling and fallbacks.

    Args:
        collection_id: Earth Engine image collection ID
        bands: List of bands to select
        reducer: Type of reducer to apply ("mean" or "sum")
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        ee.Image: Computed EVI or None if data is missing
    """
    try:
        reducer_fn = ee.Reducer.mean() if reducer == "mean" else ee.Reducer.sum()

        # Filter collection by bounds and date
        collection = ee.ImageCollection(collection_id) \
            .filterBounds(geometry) \
            .filterDate(date_range[lag], date_range["0"])

        # Check if collection is empty
        collection_size = collection.size().getInfo()
        if collection_size == 0:
            variable_name = f"{collection_id} EVI {lag}"
            logging.warning(f"Empty collection for {variable_name}")
            return ee.Image(0).rename(f"EVI_{lag}")

        # Process the collection
        return collection.select(bands) \
            .reduce(reducer_fn) \
            .rename(f"{bands[0]}_{lag}")

    except Exception as e:
        variable_name = f"{collection_id} EVI {lag}"
        logging.error(f"Error computing {variable_name}: {str(e)}")
        return ee.Image(0).rename(f"EVI_{lag}")


def calculate_vhi(image: ee.Image) -> ee.Image:
    """
    Calculate Vegetation Health Index from NDVI and TCI.

    VHI = 0.5*NDVI + 0.5*TCI

    Args:
        image: Earth Engine image with NDVI and TCI bands

    Returns:
        Earth Engine image with VHI bands added
    """
    vhi30 = image.select('NDVI_30').multiply(0.5).add(
        image.select('TCI_30').multiply(0.5)).rename('VHI_30')
    vhi60 = image.select('NDVI_60').multiply(0.5).add(
        image.select('TCI_60').multiply(0.5)).rename('VHI_60')
    vhi90 = image.select('NDVI_90').multiply(0.5).add(
        image.select('TCI_90').multiply(0.5)).rename('VHI_90')
    return image.addBands([vhi30, vhi60, vhi90])
