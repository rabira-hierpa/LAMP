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
    First attempts high-level authentication, then falls back to service account,
    finally falls back to interactive authentication.

    Raises:
        Exception: If authentication fails
    """
    # Define possible locations for credentials
    possible_credential_locations = [
        EE_CREDENTIALS_PATH,  # From config
        # Current working directory
        os.path.join(os.getcwd(), "private-key.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "private-key.json"),  # Project root
    ]

    try:
        # First try: Use the high-level ee.Initialize() function
        try:
            ee.Initialize(project='desert-locust-forcast')
            logging.info(
                "Google Earth Engine initialized successfully with default project!")
            return
        except Exception as e1:
            logging.warning(
                f"Default initialization failed: {str(e1)}. Trying authentication methods...")

        # Second try: Service account authentication
        credential_found = False
        for cred_path in possible_credential_locations:
            if os.path.exists(cred_path):
                credential_found = True
                try:
                    credentials = ee.ServiceAccountCredentials(None, cred_path)
                    ee.Initialize(credentials, project='desert-locust-forcast')
                    logging.info(
                        f"Google Earth Engine authenticated using service account credentials from {cred_path}!")
                    return
                except Exception as service_e:
                    logging.warning(
                        f"Service account authentication with {cred_path} failed: {str(service_e)}")

        # Third try: Interactive authentication
        if not credential_found:
            logging.warning(
                "No credentials file found. Attempting interactive authentication...")

        try:
            ee.Authenticate()
            ee.Initialize(project='desert-locust-forcast')
            logging.info(
                "Google Earth Engine authenticated and initialized through interactive authentication!")
            return
        except Exception as e2:
            logging.error(f"Interactive authentication failed: {str(e2)}")

        # If we got here, all methods failed
        raise Exception(
            "All Earth Engine authentication methods failed. Please check your credentials.")

    except Exception as e:
        logging.error(f"Failed to authenticate with Earth Engine: {str(e)}")
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
