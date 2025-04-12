"""
Processing package for locust data analysis.

This package handles data extraction, index calculations, and export functionality
for locust monitoring and prediction.
"""

# Import core functionality
from .export import create_export_task, start_export_task, get_task_status
from .vertical_modules.data_extraction import extract_time_lagged_data, get_missing_variables
from .vertical_modules.index_calculation import calculate_all_indices, has_critical_data_missing

# Import vertical-specific modules for direct access
from .vertical_modules.vegetation_indices import calculate_vhi
from .vertical_modules.temperature_processing import calculate_tci, calculate_tvdi
from .vertical_modules.moisture_analysis import calculate_ndwi
from .vertical_modules.wind_processing import compute_wind_components, calculate_wind_speed, calculate_wind_direction
