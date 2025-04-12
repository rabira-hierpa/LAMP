"""
Moisture analysis module for the locust processing package.

This module handles NDWI/SMAP/CHIRPS calculations and processing.
"""

import ee
import logging
from typing import Dict, List, Optional, Union

# Global reference to Ethiopia boundary - will be initialized in the main module
et_boundary = None


def set_region_boundary(boundary: ee.FeatureCollection) -> None:
    """
    Set the region boundary for moisture calculations.

    Args:
        boundary: Earth Engine feature collection defining the boundary
    """
    global et_boundary
    et_boundary = boundary


def compute_precipitation(collection_id: str, bands: List[str], reducer: str, lag: str,
                        geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> Optional[ee.Image]:
    """
    Compute precipitation from CHIRPS with error handling and fallbacks.

    Args:
        collection_id: Earth Engine image collection ID
        bands: List of bands to select
        reducer: Type of reducer to apply ("mean" or "sum")
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        ee.Image: Computed precipitation or None if data is missing
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


def compute_soil_moisture(collection_id: str, bands: List[str], reducer: str, lag: str,
                         geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> Optional[ee.Image]:
    """
    Compute soil moisture from SMAP with error handling and fallbacks.

    Args:
        collection_id: Earth Engine image collection ID
        bands: List of bands to select
        reducer: Type of reducer to apply ("mean" or "sum")
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        ee.Image: Computed soil moisture or None if data is missing
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


def compute_ndwi(lag: str, geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> ee.Image:
    """
    Calculate NDWI with fallback to approximate using NDVI and EVI.

    Args:
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        ee.Image: NDWI image
    """
    try:
        # First try to use MOD09GA
        collection = ee.ImageCollection("MODIS/006/MOD09GA") \
            .filterBounds(geometry) \
            .filterDate(date_range[lag], date_range["0"])

        collection_size = collection.size().getInfo()
        if collection_size == 0:
            logging.warning(f"Empty collection for NDWI {lag}")
            logging.info(
                f"Using approximated NDWI from MOD13Q1 (NDVI and EVI)")

            # Alternative: Approximate NDWI using NDVI and EVI from MOD13Q1
            ndvi = compute_variable(
                "MODIS/061/MOD13Q1", ["NDVI"], "mean", lag, geometry, date_range)
            evi = compute_variable("MODIS/061/MOD13Q1",
                                ["EVI"], "mean", lag, geometry, date_range)

            # Create a simple water index proxy using relationship between NDVI and EVI
            return evi.subtract(ndvi).multiply(0.5).add(0.5).rename(f'NDWI_{lag}')

        def calc_ndwi(img):
            # NDWI = (NIR - SWIR) / (NIR + SWIR)
            # For MODIS, using bands 2 (NIR) and 6 (SWIR)
            return img.normalizedDifference(['sur_refl_b02', 'sur_refl_b06']).rename('NDWI')

        ndwi_collection = collection.map(calc_ndwi)
        return ndwi_collection.mean().rename(f'NDWI_{lag}')

    except Exception as e:
        logging.error(f"Error computing NDWI for lag {lag}: {str(e)}")
        return ee.Image(0).rename(f'NDWI_{lag}')


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


def calculate_ndwi(image: ee.Image) -> ee.Image:
    """
    Calculate Normalized Difference Water Index.

    The NDWI bands should be already computed and present in the input image
    with names NDWI_30, NDWI_60, NDWI_90.

    Args:
        image: Earth Engine image with NDWI bands

    Returns:
        Earth Engine image with NDWI bands renamed/reorganized
    """
    ndwi30 = image.select('NDWI_30').rename('NDWI_30')
    ndwi60 = image.select('NDWI_60').rename('NDWI_60')
    ndwi90 = image.select('NDWI_90').rename('NDWI_90')
    return image.addBands([ndwi30, ndwi60, ndwi90])
