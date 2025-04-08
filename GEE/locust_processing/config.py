"""
Configuration settings for the locust processing package.
"""

import os

# Earth Engine dataset paths
FAO_REPORT_ASSET_ID = 'projects/desert-locust-forcast/assets/FAO_DL_data_extracted_2015'
BOUNDARIES_DATASET = "USDOS/LSIB_SIMPLE/2017"

# Export settings
COMMON_SCALE = 250
COMMON_PROJECTION = 'EPSG:4326'
EXPORT_FOLDER = 'Locust_Export'
MAX_PIXELS = 1e13

# Buffering settings
POINT_BUFFER_METERS = 5000  # 5km buffer (matching the new script)

# Service account settings
EE_CREDENTIALS_PATH = os.path.expanduser("~/private-key.json")

# Task management settings
MAX_CONCURRENT_TASKS = 240
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 300

# Progress tracking
DEFAULT_PROGRESS_FILE = 'locust_export_progress.json'
DEFAULT_LOG_FILE = 'locust_export.log'

# Processing settings
DEFAULT_BATCH_SIZE = 250

# Data specific constants
DATE_FORMAT = '%Y-%m-%d'
VALID_PRESENCE_VALUES = ['PRESENT', 'ABSENT']

# Grid settings for locust report images
GRID_SIZE = 7
SPATIAL_RESOLUTION = 1000  # 1km resolution

# Critical variables that must be present (can be overridden)
CRITICAL_VARIABLES = [
    "MODIS/061/MOD13Q1 NDVI 30",
    "MODIS/061/MOD13Q1 NDVI 60",
    "MODIS/061/MOD13Q1 NDVI 90",
    "MODIS/061/MOD11A2 LST_Day_1km 30",
    "MODIS/061/MOD11A2 LST_Day_1km 60",
    "MODIS/061/MOD11A2 LST_Day_1km 90"
]

# Alternative data sources for fallbacks
ALTERNATIVE_SOURCES = {
    "WIND": {
        "PRIMARY": "ECMWF/ERA5/DAILY",
        "FALLBACK": "NCEP_DOE_II/daily_averages"
    },
    "NDWI": {
        "PRIMARY": "MODIS/061/MOD09GA",
        "FALLBACK": "MODIS/061/MOD13Q1"  # Approximate using NDVI and EVI
    },
    "AET": {
        "PRIMARY": "MODIS/061/MOD16A2",
        "FALLBACK": "IDAHO_EPSCOR/TERRACLIMATE"
    }
}
