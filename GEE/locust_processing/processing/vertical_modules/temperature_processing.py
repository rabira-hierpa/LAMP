"""
Temperature processing module for the locust processing package.

This module handles LST/TCI/TVDI calculations and processing.
"""

import ee
import logging
from typing import Dict, List, Optional, Union

# Global reference to Ethiopia boundary - will be initialized in the main module
et_boundary = None


def set_region_boundary(boundary: ee.FeatureCollection) -> None:
    """
    Set the region boundary for temperature calculations.

    Args:
        boundary: Earth Engine feature collection defining the boundary
    """
    global et_boundary
    et_boundary = boundary


def compute_lst(collection_id: str, bands: List[str], reducer: str, lag: str,
                geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> Optional[ee.Image]:
    """
    Compute Land Surface Temperature from an image collection with error handling and fallbacks.

    Args:
        collection_id: Earth Engine image collection ID
        bands: List of bands to select
        reducer: Type of reducer to apply ("mean" or "sum")
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        ee.Image: Computed LST or None if data is missing
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
            variable_name = f"{collection_id} {bands[0]} {lag}"
            logging.warning(f"Empty collection for {variable_name}")
            return ee.Image(0).rename(f"{bands[0]}_{lag}")

        # Process the collection
        return collection.select(bands) \
            .reduce(reducer_fn) \
            .rename(f"{bands[0]}_{lag}")

    except Exception as e:
        variable_name = f"{collection_id} {bands[0]} {lag}"
        logging.error(f"Error computing {variable_name}: {str(e)}")
        return ee.Image(0).rename(f"{bands[0]}_{lag}")


def calculate_tci(image: ee.Image) -> ee.Image:
    """
    Calculate Temperature Condition Index.

    TCI = (LST - 273.15) * 0.1 (scaling to match NDVI range)

    Args:
        image: Earth Engine image with LST bands

    Returns:
        Earth Engine image with TCI bands added
    """
    tci30 = image.select('LST_Day_1km_30').subtract(
        273.15).multiply(0.1).rename('TCI_30')
    tci60 = image.select('LST_Day_1km_60').subtract(
        273.15).multiply(0.1).rename('TCI_60')
    tci90 = image.select('LST_Day_1km_90').subtract(
        273.15).multiply(0.1).rename('TCI_90')
    return image.addBands([tci30, tci60, tci90])


def calculate_tvdi(image: ee.Image) -> ee.Image:
    """
    Calculate Temperature Vegetation Dryness Index.

    TVDI = (LST - 273.15) / (NDVI * 10 + 273.15)

    Args:
        image: Earth Engine image with NDVI and LST bands

    Returns:
        Earth Engine image with TVDI bands added
    """
    # Simplified approach for calculating TVDI
    tvdi30 = image.select('LST_Day_1km_30').subtract(273.15).divide(
        image.select('NDVI_30').multiply(10).add(273.15)).rename('TVDI_30')
    tvdi60 = image.select('LST_Day_1km_60').subtract(273.15).divide(
        image.select('NDVI_60').multiply(10).add(273.15)).rename('TVDI_60')
    tvdi90 = image.select('LST_Day_1km_90').subtract(273.15).divide(
        image.select('NDVI_90').multiply(10).add(273.15)).rename('TVDI_90')

    return image.addBands([tvdi30, tvdi60, tvdi90])
