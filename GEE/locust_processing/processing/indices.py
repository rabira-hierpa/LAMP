"""
Index calculation functions for the locust processing package.
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

    Args:
        image: Earth Engine image with NDVI and LST bands

    Returns:
        Earth Engine image with TVDI bands added
    """
    global et_boundary

    if et_boundary is None:
        logging.warning(
            "Region boundary not set for TVDI calculation. Using global extent.")
        et_boundary = ee.Geometry.Rectangle([-180, -90, 180, 90])

    def compute_tvdi(ndvi, lst, ndvi_tag, lst_tag):
        # Get the LST min for the region
        lst_min = lst.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=et_boundary,
            scale=1000,
            maxPixels=1e9
        ).get(lst_tag)

        # Define parameters for the dry edge (simplified approximation)
        a = 273.15  # Intercept
        b = 50      # Slope

        # Calculate TVDI
        return lst.subtract(lst_min) \
            .divide(a.add(ndvi.multiply(b)).subtract(lst_min)) \
            .rename('TVDI_' + ndvi_tag.split('_')[-1])

    # Calculate TVDI for each time period
    tvdi30 = compute_tvdi(
        image.select('NDVI_30'),
        image.select('LST_Day_1km_30'),
        'NDVI_30',
        'LST_Day_1km_30'
    )

    tvdi60 = compute_tvdi(
        image.select('NDVI_60'),
        image.select('LST_Day_1km_60'),
        'NDVI_60',
        'LST_Day_1km_60'
    )

    tvdi90 = compute_tvdi(
        image.select('NDVI_90'),
        image.select('LST_Day_1km_90'),
        'NDVI_90',
        'LST_Day_1km_90'
    )

    return image.addBands([tvdi30, tvdi60, tvdi90])
