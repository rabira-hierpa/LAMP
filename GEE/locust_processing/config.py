"""
Configuration settings for the locust processing package.
"""

import os

# Earth Engine dataset paths
FAO_REPORT_ASSET_ID = 'projects/desert-locust-forcast/assets/FAO_filtered_data_2000'
BOUNDARIES_DATASET = "USDOS/LSIB_SIMPLE/2017"

# Export settings
COMMON_SCALE = 250
COMMON_PROJECTION = 'EPSG:4326'
EXPORT_FOLDER = 'Locust_Export'
MAX_PIXELS = 1e13

# Buffering settings
POINT_BUFFER_METERS = 5000  # 5km buffer

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
