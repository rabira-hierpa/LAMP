"""
Configuration settings for the locust processing package.

This module includes enhanced validation for Earth Engine dataset paths,
auto-loading regional boundaries, and cloud-based config override support.
"""

import os
import json
import logging
import ee
from typing import Dict, List, Optional, Any, Union, Set


class ConfigValidator:
    """
    Validator for Earth Engine dataset paths and configuration settings.
    """
    
    @staticmethod
    def validate_ee_dataset(dataset_path: str) -> bool:
        """
        Validate that an Earth Engine dataset path exists.
        
        Args:
            dataset_path: Path to an Earth Engine dataset
            
        Returns:
            bool: True if dataset exists, False otherwise
        """
        try:
            # Check if collection or image
            if '/' in dataset_path:
                # For collections, try to access info
                ee.ImageCollection(dataset_path).limit(1).getInfo()
            else:
                # For assets, check if they exist
                info = ee.data.getAsset(dataset_path)
                if info is None:
                    return False
            return True
        except ee.EEException:
            logging.warning(f"Dataset validation failed for: {dataset_path}")
            return False
    
    @staticmethod
    def validate_config(config_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate configuration settings and return validation issues.
        
        Args:
            config_dict: Dictionary of configuration settings
            
        Returns:
            Dict: Dictionary of validation issues by severity
        """
        issues = {
            "critical": [],
            "warning": [],
            "info": []
        }
        
        # Validate dataset paths
        for key, value in config_dict.items():
            if 'ASSET_ID' in key or 'DATASET' in key:
                if not ConfigValidator.validate_ee_dataset(value):
                    issues["critical"].append(
                        f"Invalid Earth Engine dataset path: {key}={value}")
        
        # Validate numeric parameters
        if config_dict.get("MAX_CONCURRENT_TASKS", 0) > 250:
            issues["warning"].append(
                "MAX_CONCURRENT_TASKS exceeds Earth Engine recommended limit of 250")
        
        # Validate buffer settings
        if config_dict.get("POINT_BUFFER_METERS", 0) > 50000:
            issues["warning"].append(
                "POINT_BUFFER_METERS exceeds 50km, which may cause processing issues")
        
        return issues


class BoundaryLoader:
    """
    Handles loading and caching of geographic boundaries from Natural Earth datasets.
    """
    
    _boundaries_cache = {}
    
    @staticmethod
    def load_country_boundary(country_code: str) -> Optional[ee.FeatureCollection]:
        """
        Load country boundary from Natural Earth dataset.
        
        Args:
            country_code: ISO 3166-1 alpha-3 country code
            
        Returns:
            ee.FeatureCollection: Feature collection for the country boundary
        """
        # Check cache first
        if country_code in BoundaryLoader._boundaries_cache:
            return BoundaryLoader._boundaries_cache[country_code]
        
        try:
            # Load from Natural Earth dataset
            countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
            country = countries.filter(ee.Filter.eq('country_co', country_code))
            
            # Verify country exists
            count = country.size().getInfo()
            if count == 0:
                logging.warning(f"Country code not found: {country_code}")
                return None
            
            # Cache and return
            BoundaryLoader._boundaries_cache[country_code] = country
            return country
        
        except ee.EEException as e:
            logging.error(f"Error loading country boundary: {str(e)}")
            return None
    
    @staticmethod
    def load_region_by_bbox(west: float, south: float, east: float, north: float) -> ee.Geometry:
        """
        Create boundary from bounding box coordinates.
        
        Args:
            west: Western longitude
            south: Southern latitude
            east: Eastern longitude
            north: Northern latitude
            
        Returns:
            ee.Geometry: Geometry for the bounding box
        """
        try:
            bbox = ee.Geometry.Rectangle([west, south, east, north])
            return bbox
        except ee.EEException as e:
            logging.error(f"Error creating bounding box: {str(e)}")
            raise


class CloudConfig:
    """
    Cloud-based configuration override system.
    """
    
    @staticmethod
    def load_from_gcs(bucket_name: str, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from Google Cloud Storage.
        
        Args:
            bucket_name: GCS bucket name
            config_path: Path to the config file in the bucket
            
        Returns:
            Dict: Configuration dictionary
        """
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(config_path)
            
            config_text = blob.download_as_text()
            return json.loads(config_text)
        
        except Exception as e:
            logging.error(f"Error loading cloud config: {str(e)}")
            return {}
    
    @staticmethod
    def save_to_gcs(config_dict: Dict[str, Any], bucket_name: str, config_path: str) -> bool:
        """
        Save configuration to Google Cloud Storage.
        
        Args:
            config_dict: Configuration dictionary
            bucket_name: GCS bucket name
            config_path: Path where to save the config in the bucket
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(config_path)
            
            config_text = json.dumps(config_dict, indent=2)
            blob.upload_from_string(config_text)
            return True
        
        except Exception as e:
            logging.error(f"Error saving cloud config: {str(e)}")
            return False


# Base configuration
# Earth Engine dataset paths
FAO_REPORT_ASSET_ID = 'projects/desert-locust-forcast/assets/FAO_DL_data_extracted_2015'
BOUNDARIES_DATASET = "USDOS/LSIB_SIMPLE/2017"

# Export settings
COMMON_SCALE = 250
COMMON_PROJECTION = 'EPSG:4326'
EXPORT_FOLDER = 'Locust_Exported_Images'
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

# Cloud configuration settings
CLOUD_CONFIG_ENABLED = False
GCS_CONFIG_BUCKET = "locust-config"
GCS_CONFIG_PATH = "config/processing_config.json"

# Update config from cloud if enabled
if CLOUD_CONFIG_ENABLED:
    try:
        cloud_config = CloudConfig.load_from_gcs(GCS_CONFIG_BUCKET, GCS_CONFIG_PATH)
        
        # Update current module globals with cloud config
        if cloud_config:
            current_globals = globals()
            for key, value in cloud_config.items():
                if key.isupper():  # Only update UPPERCASE configs
                    current_globals[key] = value
            logging.info("Configuration updated from cloud storage")
    except Exception as e:
        logging.warning(f"Failed to load cloud configuration: {str(e)}")


def validate_current_config() -> bool:
    """
    Validate the currently loaded configuration.
    
    Returns:
        bool: True if configuration is valid, False if there are critical issues
    """
    current_globals = globals()
    config_dict = {k: v for k, v in current_globals.items() 
                if k.isupper() and not k.startswith('_')}
    
    validation_issues = ConfigValidator.validate_config(config_dict)
    
    # Log all issues
    for issue in validation_issues.get("info", []):
        logging.info(f"Config info: {issue}")
        
    for issue in validation_issues.get("warning", []):
        logging.warning(f"Config warning: {issue}")
        
    for issue in validation_issues.get("critical", []):
        logging.error(f"Config error: {issue}")
    
    # Return False if there are critical issues
    return len(validation_issues.get("critical", [])) == 0
