#!/usr/bin/env python
"""
Debug script for GEE/locust_processing/cli.py

This script helps investigate the structure of Earth Engine FeatureCollections
in the locust processing pipeline.
"""

from GEE.locust_processing.config import FAO_REPORT_ASSET_ID
from GEE.locust_processing.utils.ee_utils import initialize_ee
import ee
import logging
import argparse
import os
import sys
import json

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import from locust_processing package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def safe_get_info(ee_object, default=None):
    """Get info from an EE object with error handling"""
    try:
        return ee_object.getInfo()
    except Exception as e:
        logging.error(f"Error in getInfo(): {e}")
        return default


def inspect_feature_collection(collection_id=FAO_REPORT_ASSET_ID, limit=5):
    """Inspect a feature collection and print its structure"""
    try:
        # Initialize Earth Engine
        initialize_ee()

        # Load the feature collection
        collection = ee.FeatureCollection(collection_id)
        size = collection.size().getInfo()
        logging.info(f"Collection size: {size}")

        # Get the first few features
        first_features = collection.limit(limit)
        features_info = safe_get_info(first_features)

        if not features_info or 'features' not in features_info:
            logging.error("Could not retrieve features information")
            return

        # Print feature structure
        logging.info(f"First {len(features_info['features'])} features:")

        # Print feature properties to explore structure
        for i, feature in enumerate(features_info['features']):
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            logging.info(f"\nFeature {i+1}:")
            logging.info(f"Property keys: {list(properties.keys())}")

            # Check for key properties
            for key in ['Year', 'Obs Date', 'Locust Presence', 'Longitude', 'Latitude']:
                if key in properties:
                    logging.info(f"{key}: {properties[key]}")
                else:
                    logging.info(f"{key}: Not found")

            # Check geometry
            if geometry:
                logging.info(f"Geometry type: {geometry.get('type')}")
                if 'coordinates' in geometry:
                    logging.info(f"Coordinates: {geometry['coordinates']}")
            else:
                logging.info("Geometry: Not found")

        # Test various filtering approaches
        try:
            # Filter by Year
            year_filter = collection.filter(ee.Filter.gte('Year', 2015))
            year_count = year_filter.size().getInfo()
            logging.info(f"\nFeatures with Year >= 2015: {year_count}")

            # Filter by Locust Presence
            presence_filter = collection.filter(
                ee.Filter.eq('Locust Presence', 'PRESENT'))
            presence_count = presence_filter.size().getInfo()
            logging.info(
                f"Features with Locust Presence = PRESENT: {presence_count}")

            # Test direct access
            first = collection.first()
            first_props = safe_get_info(first)
            if first_props:
                logging.info(
                    f"\nFirst feature directly: {list(first_props.get('properties', {}).keys())}")

            # Test limit and skip
            skip_limit = collection.limit(1, 2)  # Skip 2, limit 1
            skip_limit_count = skip_limit.size().getInfo()
            logging.info(f"Skip 2, limit 1 count: {skip_limit_count}")
            if skip_limit_count > 0:
                skip_limit_feature = skip_limit.first().getInfo()
                logging.info(
                    f"Feature at index 2: {list(skip_limit_feature.get('properties', {}).keys())}")

        except Exception as e:
            logging.error(f"Error testing filters: {e}")

    except Exception as e:
        logging.error(f"Error inspecting collection: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Debug Earth Engine feature collections'
    )
    parser.add_argument('--collection', type=str, default=FAO_REPORT_ASSET_ID,
                        help='Earth Engine feature collection ID')
    parser.add_argument('--limit', type=int, default=3,
                        help='Number of features to inspect')
    args = parser.parse_args()

    inspect_feature_collection(args.collection, args.limit)
