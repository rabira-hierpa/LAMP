"""
Data extraction functions for the locust processing package.
"""

import ee
import datetime
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Set

from ..utils.geo_utils import check_image_for_missing_data, create_grid_around_point, get_date_range
from ..config import SPATIAL_RESOLUTION, GRID_SIZE, FAO_REPORT_ASSET_ID, MAX_PIXELS


# Global tracker for missing data
missing_data_variables: List[str] = []


def create_locust_report_image(point: ee.Feature, date: ee.Date, spatial_resolution: int = SPATIAL_RESOLUTION) -> ee.Image:
    """
    Create a 7x7x2 image representation for locust reports.

    Args:
        point: Feature representing the center point
        date: Date for the report
        spatial_resolution: Spatial resolution in meters

    Returns:
        ee.Image: Image containing locust presence/absence counts
    """
    # Create a grid centered on the point
    grid = create_grid_around_point(point, GRID_SIZE, spatial_resolution)

    # Get previous dates for time-series analysis
    prev_date30 = date.advance(-30, 'days')
    prev_date60 = date.advance(-60, 'days')
    prev_date90 = date.advance(-90, 'days')

    # Filter locust reports for each time period
    locust_reports = ee.FeatureCollection(FAO_REPORT_ASSET_ID)

    # Count presence and absence reports for each grid cell for each time period
    def count_reports_for_period(start_date, end_date):
        period_reports = locust_reports.filterDate(start_date, end_date)

        presence_reports = period_reports.filter(
            ee.Filter.eq('Locust Presence', 'PRESENT'))
        absence_reports = period_reports.filter(
            ee.Filter.eq('Locust Presence', 'ABSENT'))

        # Count reports in each grid cell
        presence_counts = grid.map(lambda cell:
                                   cell.set('presence_count', presence_reports.filterBounds(
                                       cell.geometry()).size())
                                   )

        absence_counts = grid.map(lambda cell:
                                  cell.set('absence_count', absence_reports.filterBounds(
                                      cell.geometry()).size())
                                  )

        return {
            'presence': presence_counts,
            'absence': absence_counts
        }

    # Get counts for each time period
    counts30 = count_reports_for_period(prev_date30, date)
    counts60 = count_reports_for_period(prev_date60, prev_date30)
    counts90 = count_reports_for_period(prev_date90, prev_date60)

    # Create image representation (7x7x2) for each time period
    def create_period_image(counts, suffix):
        # Create presence image
        presence_img = ee.Image().float().paint(
            featureCollection=counts['presence'],
            color='presence_count'
        ).rename('presence_' + suffix)

        # Create absence image
        absence_img = ee.Image().float().paint(
            featureCollection=counts['absence'],
            color='absence_count'
        ).rename('absence_' + suffix)

        return ee.Image.cat([presence_img, absence_img])

    # Create images for each time period
    image30 = create_period_image(counts30, '30')
    image60 = create_period_image(counts60, '60')
    image90 = create_period_image(counts90, '90')

    return ee.Image.cat([image30, image60, image90])


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
    global missing_data_variables

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
        missing_data_variables.append(f'NDWI_{lag}')
        return ee.Image(0).rename(f'NDWI_{lag}')


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
    global missing_data_variables

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
                missing_data_variables.append(f'Wind_{lag}')
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
        missing_data_variables.append(f'Wind_{lag}')
        return {
            'u': ee.Image(0).rename(f'u_component_of_wind_10m_{lag}'),
            'v': ee.Image(0).rename(f'v_component_of_wind_10m_{lag}')
        }


def extract_time_lagged_data(point: ee.Feature) -> Optional[ee.Image]:
    """
    Extract time-lagged environmental data for a point with improved fallback mechanisms.

    Args:
        point: Feature with a parsed_date property

    Returns:
        ee.Image: Combined image with all environmental variables, or None if data is missing
    """
    global missing_data_variables
    # Reset missing data tracker for this point
    missing_data_variables = []

    # Use the parsed date we stored earlier
    date = ee.Date(point.get("parsed_date")) if point.get(
        "parsed_date") else ee.Date(datetime.datetime.now().strftime("%Y-%m-%d"))

    geometry = point.geometry()

    # Get date ranges for time periods
    date_range = {
        "0": date,  # Current date
        "30": date.advance(-30, "days"),
        "60": date.advance(-60, "days"),
        "90": date.advance(-90, "days")
    }

    # Extract static environmental data
    # Soil texture data
    soil_texture = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02") \
        .select('b0') \
        .rename('soil_texture')

    # Soil sand content at different depths
    sand_content0_5 = ee.Image("projects/soilgrids-isric/sand_mean") \
        .select('sand_0-5cm_mean') \
        .rename('sand_0_5cm')

    sand_content5_15 = ee.Image("projects/soilgrids-isric/sand_mean") \
        .select('sand_5-15cm_mean') \
        .rename('sand_5_15cm')

    sand_content15_30 = ee.Image("projects/soilgrids-isric/sand_mean") \
        .select('sand_15-30cm_mean') \
        .rename('sand_15_30cm')

    # Add elevation data (SRTM)
    elevation = ee.Image('USGS/SRTMGL1_003') \
        .rename('elevation')

    # Derive slope and aspect
    terrain = ee.Algorithms.Terrain(elevation)
    slope = terrain.select('slope').rename('slope')
    aspect = terrain.select('aspect').rename('aspect')

    # Add land cover
    landcover = ee.Image("MODIS/006/MCD12Q1/2019_01_01") \
        .select('LC_Type1') \
        .rename('landcover')

    # Calculate dynamic variables
    # NDVI for each time period
    ndvi30 = compute_variable(
        "MODIS/061/MOD13Q1", ["NDVI"], "mean", "30", geometry, date_range)
    ndvi60 = compute_variable(
        "MODIS/061/MOD13Q1", ["NDVI"], "mean", "60", geometry, date_range)
    ndvi90 = compute_variable(
        "MODIS/061/MOD13Q1", ["NDVI"], "mean", "90", geometry, date_range)

    # EVI for each time period
    evi30 = compute_variable(
        "MODIS/061/MOD13Q1", ["EVI"], "mean", "30", geometry, date_range)
    evi60 = compute_variable(
        "MODIS/061/MOD13Q1", ["EVI"], "mean", "60", geometry, date_range)
    evi90 = compute_variable(
        "MODIS/061/MOD13Q1", ["EVI"], "mean", "90", geometry, date_range)

    # LST for each time period
    lst30 = compute_variable(
        "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "30", geometry, date_range)
    lst60 = compute_variable(
        "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "60", geometry, date_range)
    lst90 = compute_variable(
        "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "90", geometry, date_range)

    # Precipitation for each time period
    precip30 = compute_variable(
        "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "30", geometry, date_range)
    precip60 = compute_variable(
        "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "60", geometry, date_range)
    precip90 = compute_variable(
        "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "90", geometry, date_range)

    # Wind components for each time period
    wind30 = compute_wind_components("30", geometry, date_range)
    wind60 = compute_wind_components("60", geometry, date_range)
    wind90 = compute_wind_components("90", geometry, date_range)

    # Soil moisture for each time period
    soil_moisture30 = compute_variable(
        "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "30", geometry, date_range)
    soil_moisture60 = compute_variable(
        "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "60", geometry, date_range)
    soil_moisture90 = compute_variable(
        "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "90", geometry, date_range)

    # Actual Evapotranspiration for each time period
    aet30 = compute_variable(
        "MODIS/006/MOD16A2", ["ET"], "sum", "30", geometry, date_range)
    aet60 = compute_variable(
        "MODIS/006/MOD16A2", ["ET"], "sum", "60", geometry, date_range)
    aet90 = compute_variable(
        "MODIS/006/MOD16A2", ["ET"], "sum", "90", geometry, date_range)

    # TerraClimate AET for each time period (as fallback to MODIS AET)
    terra_aet30 = compute_variable(
        "IDAHO_EPSCOR/TERRACLIMATE", ["aet"], "sum", "30", geometry, date_range)
    terra_aet60 = compute_variable(
        "IDAHO_EPSCOR/TERRACLIMATE", ["aet"], "sum", "60", geometry, date_range)
    terra_aet90 = compute_variable(
        "IDAHO_EPSCOR/TERRACLIMATE", ["aet"], "sum", "90", geometry, date_range)

    # TerraClimate PET for each time period
    terra_pet30 = compute_variable(
        "IDAHO_EPSCOR/TERRACLIMATE", ["pet"], "mean", "30", geometry, date_range)
    terra_pet60 = compute_variable(
        "IDAHO_EPSCOR/TERRACLIMATE", ["pet"], "mean", "60", geometry, date_range)
    terra_pet90 = compute_variable(
        "IDAHO_EPSCOR/TERRACLIMATE", ["pet"], "mean", "90", geometry, date_range)

    # NDWI for each time period (with fallbacks)
    ndwi30 = compute_ndwi("30", geometry, date_range)
    ndwi60 = compute_ndwi("60", geometry, date_range)
    ndwi90 = compute_ndwi("90", geometry, date_range)

    # Check if any critical variables are missing
    if has_critical_data_missing():
        logging.warning(
            "Critical data is missing. Cannot proceed with image creation.")
        return None

    # Combine all environmental variables
    static_variables = [
        elevation, slope, aspect,
        landcover, soil_texture,
        sand_content0_5, sand_content5_15, sand_content15_30
    ]

    dynamic_variables = [
        ndvi30, ndvi60, ndvi90,
        evi30, evi60, evi90,
        lst30, lst60, lst90,
        precip30, precip60, precip90,
        wind30['u'], wind60['u'], wind90['u'],
        wind30['v'], wind60['v'], wind90['v'],
        soil_moisture30, soil_moisture60, soil_moisture90,
        aet30, aet60, aet90,
        terra_aet30, terra_aet60, terra_aet90,
        terra_pet30, terra_pet60, terra_pet90,
        ndwi30, ndwi60, ndwi90
    ]

    # Import indices calculation functions
    from ..processing.indices import calculate_all_indices

    # Combine all variables into a single image
    combined_image = ee.Image.cat(dynamic_variables + static_variables)

    # Calculate indices
    combined_image = calculate_all_indices(combined_image)

    # Create 7x7x2 locust report image
    locust_report_image = create_locust_report_image(
        point, date, SPATIAL_RESOLUTION)

    # Combine all data
    final_image = ee.Image.cat([combined_image, locust_report_image])

    # Check for missing data in the final image
    if check_image_for_missing_data(final_image, geometry.buffer(10000)):
        logging.warning(
            "Final image has missing data but critical bands are present")
        # Continue since we've already checked for critical bands

    return final_image


def has_critical_data_missing() -> bool:
    """
    Check if critical variables are missing from the extracted data.

    Returns:
        bool: True if critical data is missing, False otherwise
    """
    global missing_data_variables

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
        if var in missing_data_variables:
            logging.error(f"Critical variable missing: {var}")
            return True

    return False


def get_missing_variables() -> List[str]:
    """
    Get the list of missing variables for the current extraction.

    Returns:
        List[str]: List of missing variable names
    """
    global missing_data_variables
    return missing_data_variables


def verify_presence_value(feature: ee.Feature, feature_index: int) -> Optional[str]:
    """
    Verify the locust presence value from a feature.

    Args:
        feature: Feature containing 'Locust Presence' property
        feature_index: Index for logging purposes

    Returns:
        Presence value string ('PRESENT' or 'ABSENT') if valid, None otherwise
    """
    try:
        has_presence = feature.propertyNames().contains('Locust Presence')
        presence_raw = ee.Algorithms.If(
            has_presence,
            feature.get('Locust Presence'),
            None  # Default to None if property doesn't exist
        )
        presence_is_null = ee.Algorithms.IsEqual(presence_raw, None)

        # Check if presence is null server-side
        if presence_is_null.getInfo():
            logging.warning(
                f"Feature {feature_index} has missing or null 'Locust Presence'. Skipping.")
            return None

        presence = presence_raw.getInfo()  # GetInfo only when needed

        # Validate presence value client-side
        if presence not in ['PRESENT', 'ABSENT']:
            logging.warning(
                f"Feature {feature_index} has invalid presence value: '{presence}'. Skipping.")
            return None

        logging.info(
            f"Feature {feature_index}: Presence value '{presence}' is valid.")
        return presence

    except Exception as e:
        logging.warning(
            f"Error getting presence for feature {feature_index}: {e}. Skipping.")
        return None
