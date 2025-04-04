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


def calculate_all_indices(image: ee.Image) -> ee.Image:
    """
    Calculate all vegetation and drought indices.

    Args:
        image: Earth Engine image with required bands

    Returns:
        Earth Engine image with all indices added
    """
    image = calculate_tci(image)
    image = calculate_vhi(image)
    image = calculate_ndwi(image)
    image = calculate_tvdi(image)
    return image
