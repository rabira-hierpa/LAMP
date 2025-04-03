"""
Earth Engine utilities for the locust processing package.
"""

import os
import logging
import ee
from typing import Optional

from ..config import EE_CREDENTIALS_PATH


def initialize_ee() -> None:
    """
    Initialize Earth Engine with authentication.
    First attempts high-level authentication, then falls back to service account.

    Raises:
        Exception: If authentication fails
    """
    try:
        # Try using the high-level ee.Authenticate() function
        ee.Authenticate()
        ee.Initialize()
        logging.info(
            "Google Earth Engine authenticated and initialized successfully!")
    except Exception as e:
        # If the high-level function fails, try service account authentication
        try:
            # Check if credentials file exists
            if not os.path.exists(EE_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Credentials file not found at {EE_CREDENTIALS_PATH}")

            # Initialize with service account
            credentials = ee.ServiceAccountCredentials(
                None, EE_CREDENTIALS_PATH)
            ee.Initialize(credentials)
            logging.info(
                "Google Earth Engine authenticated using service account credentials!")
        except Exception as sub_e:
            logging.error(
                f"Failed to authenticate with Earth Engine: {str(sub_e)}")
            raise


def check_ee_initialized() -> bool:
    """
    Check if Earth Engine has been initialized.

    Returns:
        bool: True if initialized, False otherwise
    """
    try:
        ee.Number(1).getInfo()
        return True
    except:
        return False


def ensure_ee_initialized() -> None:
    """
    Ensure Earth Engine is initialized before proceeding.
    If not already initialized, attempts to initialize it.

    Raises:
        RuntimeError: If initialization fails
    """
    if not check_ee_initialized():
        try:
            initialize_ee()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Earth Engine: {str(e)}")

    if not check_ee_initialized():
        raise RuntimeError(
            "Earth Engine initialization check failed even after initialization attempt")


def add_cumulative_index():
    """
    Returns a function that adds cumulative index to a feature collection.

    Returns:
        function: A function that takes a FeatureCollection and returns an 
                  indexed version of the collection
    """
    # Accumulator to keep track of index
    def add_index_iter(feature, acc):
        acc = ee.Dictionary(acc)
        # Cast feature to ee.Feature before setting property
        feature = ee.Feature(feature)
        index = ee.Number(acc.get('index')).add(1)
        features = ee.List(acc.get('features'))

        return ee.Dictionary({
            'features': features.add(feature.set('index', index)),
            'index': index
        })

    # Return a function that takes a collection as input
    def process_collection(collection):
        accumulator = ee.Dictionary({'features': ee.List([]), 'index': -1})
        # Ensure the input is a FeatureCollection
        collection = ee.FeatureCollection(collection)
        result = collection.iterate(add_index_iter, accumulator)
        return ee.FeatureCollection(ee.List(ee.Dictionary(result).get('features')))

    return process_collection
