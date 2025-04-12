"""
Batch management module for optimized feature processing.

This module provides batch-aware memory management and spatial partitioning
for large Earth Engine exports.
"""

import ee
import gc
import logging
import math
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterator, Union
from functools import partial

from ...config import DEFAULT_BATCH_SIZE, MAX_PIXELS


class BatchManager:
    """
    Manages batched processing of features with memory optimization.
    """
    
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initialize batch manager.
        
        Args:
            batch_size: Number of features to process in a single batch
        """
        self.batch_size = batch_size
        self.processed_count = 0
        self.memory_usage = []  # Track memory usage over time
        
    def create_batches(self, feature_collection: ee.FeatureCollection) -> List[ee.FeatureCollection]:
        """
        Split a feature collection into batches for efficient processing.
        
        Args:
            feature_collection: Earth Engine feature collection to split
            
        Returns:
            List of feature collection batches
        """
        total_features = feature_collection.size().getInfo()
        batches = []
        
        for i in range(0, total_features, self.batch_size):
            end_idx = min(i + self.batch_size, total_features)
            batch = feature_collection.toList(end_idx).slice(i)
            batches.append(ee.FeatureCollection(batch))
            
        logging.info(f"Created {len(batches)} batches of size {self.batch_size}")
        return batches
    
    def process_in_batches(self, 
                          feature_collection: ee.FeatureCollection, 
                          process_func: Callable[[ee.Feature, int], Any], 
                          show_progress: bool = True) -> List[Any]:
        """
        Process features in batches with memory management.
        
        Args:
            feature_collection: Features to process
            process_func: Function to apply to each feature
            show_progress: Whether to show progress messages
            
        Returns:
            List of results from processing each feature
        """
        batches = self.create_batches(feature_collection)
        results = []
        batch_count = len(batches)
        
        for batch_idx, batch in enumerate(batches):
            if show_progress:
                logging.info(f"Processing batch {batch_idx+1}/{batch_count}")
            
            # Convert batch to Python list for local processing
            features = batch.toList(self.batch_size).getInfo()
            batch_results = []
            
            for feature_idx, feature_dict in enumerate(features):
                global_idx = batch_idx * self.batch_size + feature_idx
                
                # Process feature and collect result
                feature_ee = ee.Feature(feature_dict)
                try:
                    result = process_func(feature_ee, global_idx)
                    batch_results.append(result)
                    self.processed_count += 1
                    
                    # Log progress periodically
                    if self.processed_count % 50 == 0 and show_progress:
                        logging.info(f"Processed {self.processed_count} features")
                        
                except Exception as e:
                    logging.error(f"Error processing feature {global_idx}: {str(e)}")
                    batch_results.append(None)
            
            # Add batch results to overall results
            results.extend(batch_results)
            
            # Perform memory cleanup after each batch
            self._memory_cleanup()
            
        return results
    
    def _memory_cleanup(self):
        """
        Perform memory cleanup operations between batches.
        """
        # Force garbage collection
        gc.collect()
        
        # Record current memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_usage.append(memory_mb)
            
            if len(self.memory_usage) > 1:
                diff = memory_mb - self.memory_usage[-2]
                if diff > 100:  # If memory increased by more than 100 MB
                    logging.warning(f"Memory usage increased by {diff:.2f} MB since last batch")
        except ImportError:
            # psutil not available, skip memory tracking
            pass


class SpatialPartitioner:
    """
    Partitions large regions into manageable tiles for efficient processing.
    """
    
    def __init__(self, 
                max_pixels: int = MAX_PIXELS, 
                scale: int = 1000):
        """
        Initialize spatial partitioner.
        
        Args:
            max_pixels: Maximum number of pixels per partition
            scale: Scale in meters to use for calculations
        """
        self.max_pixels = max_pixels
        self.scale = scale
        
    def partition_geometry(self, geometry: ee.Geometry) -> List[ee.Geometry]:
        """
        Partition a geometry into manageable tiles.
        
        Args:
            geometry: Earth Engine geometry to partition
            
        Returns:
            List of geometry tiles
        """
        # Get geometry area
        area_m2 = geometry.area().getInfo()
        
        # Calculate pixel area
        pixel_area_m2 = self.scale * self.scale
        
        # Estimate number of pixels
        estimated_pixels = area_m2 / pixel_area_m2
        
        if estimated_pixels <= self.max_pixels:
            # Small enough to process as a single tile
            return [geometry]
        
        # Need to partition - determine how many tiles needed
        num_tiles = math.ceil(estimated_pixels / self.max_pixels)
        
        # Determine grid dimensions (trying to keep tiles square-ish)
        grid_size = math.ceil(math.sqrt(num_tiles))
        
        # Get bounds
        bounds = geometry.bounds().getInfo()['coordinates'][0]
        xs = [p[0] for p in bounds]
        ys = [p[1] for p in bounds]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Calculate tile sizes
        tile_width = (max_x - min_x) / grid_size
        tile_height = (max_y - min_y) / grid_size
        
        # Create tiles
        tiles = []
        for i in range(grid_size):
            for j in range(grid_size):
                x_start = min_x + i * tile_width
                y_start = min_y + j * tile_height
                x_end = min_x + (i + 1) * tile_width
                y_end = min_y + (j + 1) * tile_height
                
                tile = ee.Geometry.Rectangle([x_start, y_start, x_end, y_end])
                # Only include tiles that intersect the original geometry
                if geometry.intersects(tile).getInfo():
                    tiles.append(tile.intersection(geometry))
        
        logging.info(f"Partitioned geometry into {len(tiles)} tiles")
        return tiles
    
    def process_partitions(self, 
                          geometry: ee.Geometry, 
                          process_func: Callable[[ee.Geometry, int], Any],
                          combine_func: Callable[[List[Any]], Any] = None) -> Any:
        """
        Process a large geometry by partitioning it into manageable parts.
        
        Args:
            geometry: Large geometry to partition and process
            process_func: Function to apply to each partition
            combine_func: Function to combine partition results (optional)
            
        Returns:
            Combined results if combine_func is provided, otherwise list of results
        """
        partitions = self.partition_geometry(geometry)
        results = []
        
        for idx, partition in enumerate(partitions):
            logging.info(f"Processing partition {idx+1}/{len(partitions)}")
            try:
                result = process_func(partition, idx)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing partition {idx}: {str(e)}")
        
        # Combine results if a combine function is provided
        if combine_func and results:
            return combine_func(results)
        
        return results


class FAOReportValidator:
    """
    Validates FAO locust report data for quality and correctness.
    """
    
    def __init__(self):
        """
        Initialize FAO report validator.
        """
        self.validation_errors = []
        self.valid_fields = [
            'Locust Presence', 'FINISHDATE', 'STARTDATE', 
            'Development Stage', 'Population Type'
        ]
        self.required_fields = ['Locust Presence', 'FINISHDATE']
        self.valid_presence_values = ['PRESENT', 'ABSENT']
    
    def validate_report(self, feature_collection: ee.FeatureCollection) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a FAO locust report feature collection.
        
        Args:
            feature_collection: Earth Engine feature collection of FAO reports
            
        Returns:
            Tuple of (is_valid, validation_stats)
        """
        self.validation_errors = []
        
        # Check if collection is empty
        feature_count = feature_collection.size().getInfo()
        if feature_count == 0:
            self.validation_errors.append("Feature collection is empty")
            return False, {"error_count": 1, "errors": self.validation_errors}
        
        # Get a sample of features for validation
        sample_size = min(100, feature_count)
        sample = feature_collection.toList(sample_size).getInfo()
        
        # Check required fields
        missing_fields = {}
        invalid_values = {}
        
        for feature_idx, feature_dict in enumerate(sample):
            feature = ee.Feature(feature_dict)
            properties = feature.propertyNames().getInfo()
            
            # Check each required field
            for field in self.required_fields:
                if field not in properties:
                    if field not in missing_fields:
                        missing_fields[field] = []
                    missing_fields[field].append(feature_idx)
                elif field == 'Locust Presence':
                    # Special validation for locust presence
                    presence = feature.get(field).getInfo()
                    if presence not in self.valid_presence_values:
                        if field not in invalid_values:
                            invalid_values[field] = []
                        invalid_values[field].append((feature_idx, presence))
        
        # Add errors to validation errors list
        for field, indices in missing_fields.items():
            self.validation_errors.append(
                f"Field '{field}' missing in {len(indices)} features ({', '.join(map(str, indices[:10]))}...)")
        
        for field, values in invalid_values.items():
            self.validation_errors.append(
                f"Field '{field}' has invalid values in {len(values)} features")
        
        # Prepare validation stats
        validation_stats = {
            "feature_count": feature_count,
            "sample_size": sample_size,
            "missing_fields": missing_fields,
            "invalid_values": invalid_values,
            "error_count": len(self.validation_errors),
            "errors": self.validation_errors
        }
        
        # Determine if the report is valid for processing
        is_valid = len(self.validation_errors) == 0
        
        return is_valid, validation_stats
    
    def filter_valid_features(self, feature_collection: ee.FeatureCollection) -> ee.FeatureCollection:
        """
        Filter a feature collection to only include valid features.
        
        Args:
            feature_collection: Earth Engine feature collection of FAO reports
            
        Returns:
            Filtered feature collection with only valid features
        """
        # Filter for features with required properties
        for field in self.required_fields:
            feature_collection = feature_collection.filter(ee.Filter.notNull([field]))
        
        # Filter for valid presence values
        presence_filter = ee.Filter.inList('Locust Presence', self.valid_presence_values)
        feature_collection = feature_collection.filter(presence_filter)
        
        return feature_collection
