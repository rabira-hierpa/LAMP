"""
Wind processing module for the locust processing package.

This module handles ERA5 wind components and processing.
"""

import ee
import logging
from typing import Dict, List, Optional, Union

# Global reference to Ethiopia boundary - will be initialized in the main module
et_boundary = None


def set_region_boundary(boundary: ee.FeatureCollection) -> None:
    """
    Set the region boundary for wind calculations.

    Args:
        boundary: Earth Engine feature collection defining the boundary
    """
    global et_boundary
    et_boundary = boundary


def compute_wind_components(lag: str, geometry: ee.Geometry, date_range: Dict[str, ee.Date]) -> Dict[str, ee.Image]:
    """
    Calculate wind components with fallback to alternative data source.

    Args:
        lag: Time lag key ("30", "60", or "90")
        geometry: Geometry to filter by
        date_range: Dictionary of date ranges

    Returns:
        Dict: Dictionary with 'u' and 'v' wind component images
    """
    try:
        # Try ERA5 first (preferred source)
        era5_collection = ee.ImageCollection("ECMWF/ERA5/DAILY") \
            .filterBounds(geometry) \
            .filterDate(date_range[lag], date_range["0"])

        era5_size = era5_collection.size().getInfo()
        if era5_size == 0:
            logging.warning(
                f"ERA5 wind data not available for lag {lag}, using NCEP/NCAR Reanalysis")

            # Alternative source: NCEP/NCAR Reanalysis
            ncep_collection = ee.ImageCollection("NCEP_DOE_II/daily_averages") \
                .filterBounds(geometry) \
                .filterDate(date_range[lag], date_range["0"])

            # Check if NCEP data is available
            ncep_size = ncep_collection.size().getInfo()
            if ncep_size == 0:
                logging.warning(f"No wind data available for lag {lag}")
                return {
                    'u': ee.Image(0).rename(f'u_component_of_wind_10m_{lag}'),
                    'v': ee.Image(0).rename(f'v_component_of_wind_10m_{lag}')
                }

            # Process NCEP data
            u_wind = ncep_collection.select(['uwnd_10m']).mean().rename(
                f'u_component_of_wind_10m_{lag}')
            v_wind = ncep_collection.select(['vwnd_10m']).mean().rename(
                f'v_component_of_wind_10m_{lag}')

            return {
                'u': u_wind,
                'v': v_wind
            }

        # Process ERA5 data
        u_wind = era5_collection.select(['u_component_of_wind_10m']).mean().rename(
            f'u_component_of_wind_10m_{lag}')
        v_wind = era5_collection.select(['v_component_of_wind_10m']).mean().rename(
            f'v_component_of_wind_10m_{lag}')

        return {
            'u': u_wind,
            'v': v_wind
        }

    except Exception as e:
        logging.error(
            f"Error computing wind components for lag {lag}: {str(e)}")
        return {
            'u': ee.Image(0).rename(f'u_component_of_wind_10m_{lag}'),
            'v': ee.Image(0).rename(f'v_component_of_wind_10m_{lag}')
        }


def calculate_wind_speed(image: ee.Image) -> ee.Image:
    """
    Calculate wind speed from wind components.
    
    Wind speed = sqrt(u^2 + v^2)

    Args:
        image: Earth Engine image with u and v wind components

    Returns:
        Earth Engine image with wind speed bands added
    """
    # Calculate wind speed for each lag period
    wind_speed_30 = image.select(f'u_component_of_wind_10m_30').pow(2) \
        .add(image.select(f'v_component_of_wind_10m_30').pow(2)) \
        .sqrt() \
        .rename('wind_speed_30')
        
    wind_speed_60 = image.select(f'u_component_of_wind_10m_60').pow(2) \
        .add(image.select(f'v_component_of_wind_10m_60').pow(2)) \
        .sqrt() \
        .rename('wind_speed_60')
        
    wind_speed_90 = image.select(f'u_component_of_wind_10m_90').pow(2) \
        .add(image.select(f'v_component_of_wind_10m_90').pow(2)) \
        .sqrt() \
        .rename('wind_speed_90')
        
    return image.addBands([wind_speed_30, wind_speed_60, wind_speed_90])


def calculate_wind_direction(image: ee.Image) -> ee.Image:
    """
    Calculate wind direction from wind components.
    
    Wind direction = atan2(v, u) * 180/π

    Args:
        image: Earth Engine image with u and v wind components

    Returns:
        Earth Engine image with wind direction bands added
    """
    # Calculate wind direction in degrees (0-360) for each lag period
    wind_dir_30 = ee.Image.constant(Math.PI) \
        .multiply(2) \
        .add(image.select(f'v_component_of_wind_10m_30').atan2(image.select(f'u_component_of_wind_10m_30'))) \
        .multiply(180).divide(Math.PI) \
        .mod(360) \
        .rename('wind_direction_30')
        
    wind_dir_60 = ee.Image.constant(Math.PI) \
        .multiply(2) \
        .add(image.select(f'v_component_of_wind_10m_60').atan2(image.select(f'u_component_of_wind_10m_60'))) \
        .multiply(180).divide(Math.PI) \
        .mod(360) \
        .rename('wind_direction_60')
        
    wind_dir_90 = ee.Image.constant(Math.PI) \
        .multiply(2) \
        .add(image.select(f'v_component_of_wind_10m_90').atan2(image.select(f'u_component_of_wind_10m_90'))) \
        .multiply(180).divide(Math.PI) \
        .mod(360) \
        .rename('wind_direction_90')
        
    return image.addBands([wind_dir_30, wind_dir_60, wind_dir_90])
