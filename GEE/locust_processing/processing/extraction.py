"""
Data extraction functions for the locust processing package.
"""

import ee
import datetime
import logging
import math
from typing import Dict, List, Optional, Union, Any

from ..utils.geo_utils import check_image_for_missing_data, create_grid_around_point, get_date_range
from ..config import SPATIAL_RESOLUTION, GRID_SIZE, FAO_REPORT_ASSET_ID


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


def extract_time_lagged_data(point: ee.Feature) -> Optional[ee.Image]:
    """
    Extract time-lagged environmental data for a point.

    Args:
        point: Feature with a parsed_date property

    Returns:
        ee.Image: Combined image with all environmental variables, or None if data is missing
    """
    # Use the parsed date we stored earlier
    date = ee.Date(point.get("parsed_date")) if point.get(
        "parsed_date") else ee.Date(datetime.datetime.now().strftime("%Y-%m-%d"))

    geometry = point.geometry()

    # Get date ranges for time periods
    lags = get_date_range(date, 90)

    def compute_variable(collection_id, bands, reducer, lag):
        reducer_fn = ee.Reducer.mean() if reducer == "mean" else ee.Reducer.sum()

        # Check if collection has data for the time period
        collection = ee.ImageCollection(collection_id) \
            .filterBounds(geometry) \
            .filterDate(lags[lag], date)

        # Skip if collection is empty
        collection_size = collection.size().getInfo()  # Get size once
        if collection_size == 0:
            logging.warning(
                f"No data available for {collection_id} with bands {bands} from {lags[lag].format().getInfo()} to {date.format().getInfo()}")
            return None

        return collection.select(bands) \
            .reduce(reducer_fn) \
            .rename(bands[0] + "_" + lag)

    # Extract soil sand content at different depths
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

    # Add land cover
    landcover = ee.Image("MODIS/006/MCD12Q1/2019_01_01") \
        .select('LC_Type1') \
        .rename('landcover')

    # Extract all the dynamic variables
    variables = {
        "MODIS/061/MOD13A2_NDVI_30": compute_variable("MODIS/061/MOD13A2", ["NDVI"], "mean", "30"),
        "MODIS/061/MOD13A2_NDVI_60": compute_variable("MODIS/061/MOD13A2", ["NDVI"], "mean", "60"),
        "MODIS/061/MOD13A2_NDVI_90": compute_variable("MODIS/061/MOD13A2", ["NDVI"], "mean", "90"),
        "MODIS/061/MOD13A2_EVI_30": compute_variable("MODIS/061/MOD13A2", ["EVI"], "mean", "30"),
        "MODIS/061/MOD13A2_EVI_60": compute_variable("MODIS/061/MOD13A2", ["EVI"], "mean", "60"),
        "MODIS/061/MOD13A2_EVI_90": compute_variable("MODIS/061/MOD13A2", ["EVI"], "mean", "90"),
        "MODIS/061/MOD11A2_LST_30": compute_variable("MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "30"),
        "MODIS/061/MOD11A2_LST_60": compute_variable("MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "60"),
        "MODIS/061/MOD11A2_LST_90": compute_variable("MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "90"),
        "CHIRPS_PRECIP_30": compute_variable("UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "30"),
        "CHIRPS_PRECIP_60": compute_variable("UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "60"),
        "CHIRPS_PRECIP_90": compute_variable("UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "90"),
        "ERA5_U_WIND_10M_30": compute_variable("ECMWF/ERA5/DAILY", ["u_component_of_wind_10m"], "mean", "30"),
        "ERA5_U_WIND_10M_60": compute_variable("ECMWF/ERA5/DAILY", ["u_component_of_wind_10m"], "mean", "60"),
        "ERA5_U_WIND_10M_90": compute_variable("ECMWF/ERA5/DAILY", ["u_component_of_wind_10m"], "mean", "90"),
        "ERA5_V_WIND_10M_30": compute_variable("ECMWF/ERA5/DAILY", ["v_component_of_wind_10m"], "mean", "30"),
        "ERA5_V_WIND_10M_60": compute_variable("ECMWF/ERA5/DAILY", ["v_component_of_wind_10m"], "mean", "60"),
        "ERA5_V_WIND_10M_90": compute_variable("ECMWF/ERA5/DAILY", ["v_component_of_wind_10m"], "mean", "90"),
        "SMAP_SM_30": compute_variable("NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "30"),
        "SMAP_SM_60": compute_variable("NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "60"),
        "SMAP_SM_90": compute_variable("NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "90"),
        "MOD16A2_AET_30": compute_variable("MODIS/006/MOD16A2", ["ET"], "sum", "30"),
        "MOD16A2_AET_60": compute_variable("MODIS/006/MOD16A2", ["ET"], "sum", "60"),
        "MOD16A2_AET_90": compute_variable("MODIS/006/MOD16A2", ["ET"], "sum", "90"),
        "TERRACLIMATE_AET_30": compute_variable("IDAHO_EPSCOR/TERRACLIMATE", ["aet"], "sum", "30"),
        "TERRACLIMATE_AET_60": compute_variable("IDAHO_EPSCOR/TERRACLIMATE", ["aet"], "sum", "60"),
        "TERRACLIMATE_AET_90": compute_variable("IDAHO_EPSCOR/TERRACLIMATE", ["aet"], "sum", "90"),
        "TERRACLIMATE_TBP_30": compute_variable("IDAHO_EPSCOR/TERRACLIMATE", ["pet"], "mean", "30"),
        "TERRACLIMATE_TBP_60": compute_variable("IDAHO_EPSCOR/TERRACLIMATE", ["pet"], "mean", "60"),
        "TERRACLIMATE_TBP_90": compute_variable("IDAHO_EPSCOR/TERRACLIMATE", ["pet"], "mean", "90"),
    }

    # Check if any of the dynamic variables are missing
    missing_variables = [name for name,
                         var in variables.items() if var is None]
    if missing_variables:
        logging.warning(
            f"Missing data for variables: {', '.join(missing_variables)}")
        # Save this to the progress indicating the index for this feature has been skipped
        save_progress(point.get('index'), 'skipped')
        return None

    # If we have all variables, continue with image creation
    images = list(variables.values()) + [
        elevation,
        landcover,
        sand_content0_5,
        sand_content5_15,
        sand_content15_30
    ]

    # Combine all imagery
    from ..processing.indices import calculate_tci, calculate_vhi

    combined_image = ee.Image.cat(images)
    combined_image = calculate_tci(combined_image)
    combined_image = calculate_vhi(combined_image)

    # Create 7x7x2 locust report image
    locust_report_image = create_locust_report_image(
        point, date, SPATIAL_RESOLUTION)

    # Combine all data
    final_image = ee.Image.cat([combined_image, locust_report_image])

    # Check for missing data in the final image
    if check_image_for_missing_data(final_image, geometry.buffer(10000)):
        logging.warning("Final image has missing data")
        return None

    return final_image


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
