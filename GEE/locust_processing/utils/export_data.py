import ee
import os
import datetime
import time
import math
import logging
import argparse
import json
import sys
from queue import Queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor

# Set up logging


def setup_logging(log_file='locust_export.log'):
    """Configure logging to both console and file"""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

# Initialize Earth Engine


def initialize_ee():
    """Initialize Earth Engine with authentication"""
    try:
        # Try using the high-level ee.Authenticate() function
        ee.Authenticate()
        ee.Initialize()
        logging.info(
            "Google Earth Engine authenticated and initialized successfully!")
    except Exception as e:
        # If the high-level function fails, try service account authentication
        try:
            # Path to your service account credentials file
            credentials_path = os.path.expanduser("~/private-key.json")
            # Check if credentials file exists
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Credentials file not found at {credentials_path}")

            # Initialize with service account
            credentials = ee.ServiceAccountCredentials(None, credentials_path)
            ee.Initialize(credentials)
            logging.info(
                "Google Earth Engine authenticated using service account credentials!")
        except Exception as sub_e:
            logging.error(
                f"Failed to authenticate with Earth Engine: {str(sub_e)}")
            raise

# Define the Ethiopia boundary


def get_ethiopia_boundary():
    """Get Ethiopia boundary as a Feature"""
    return ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', 'Ethiopia'))

# Function to calculate VHI


def calculate_vhi(image):
    """Calculate Vegetation Health Index from NDVI and TCI"""
    vhi30 = image.select('NDVI_30').multiply(0.5).add(
        image.select('TCI_30').multiply(0.5)).rename('VHI_30')
    vhi60 = image.select('NDVI_60').multiply(0.5).add(
        image.select('TCI_60').multiply(0.5)).rename('VHI_60')
    vhi90 = image.select('NDVI_90').multiply(0.5).add(
        image.select('TCI_90').multiply(0.5)).rename('VHI_90')
    return image.addBands([vhi30, vhi60, vhi90])

# Function to calculate TCI


def calculate_tci(image):
    """Calculate Temperature Condition Index"""
    tci30 = image.select('LST_Day_1km_30').subtract(
        273.15).multiply(0.1).rename('TCI_30')
    tci60 = image.select('LST_Day_1km_60').subtract(
        273.15).multiply(0.1).rename('TCI_60')
    tci90 = image.select('LST_Day_1km_90').subtract(
        273.15).multiply(0.1).rename('TCI_90')
    return image.addBands([tci30, tci60, tci90])

# Function to calculate TVDI


def calculate_tvdi(image):
    """Calculate Temperature Vegetation Dryness Index"""
    def compute_tvdi(ndvi, lst, ndvi_tag, lst_tag):
        # Get the LST min for the region
        lst_min = lst.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=et_boundary,
            scale=1000,
            maxPixels=1e9
        ).get(lst_tag)

        # Define parameters for the dry edge (simplified approximation)
        a = 273.15  # Intercept
        b = 50      # Slope

        # Calculate TVDI
        return lst.subtract(lst_min) \
            .divide(a.add(ndvi.multiply(b)).subtract(lst_min)) \
            .rename('TVDI_' + ndvi_tag.split('_')[-1])

    # Calculate TVDI for each time period
    tvdi30 = compute_tvdi(
        image.select('NDVI_30'),
        image.select('LST_Day_1km_30'),
        'NDVI_30',
        'LST_Day_1km_30'
    )

    tvdi60 = compute_tvdi(
        image.select('NDVI_60'),
        image.select('LST_Day_1km_60'),
        'NDVI_60',
        'LST_Day_1km_60'
    )

    tvdi90 = compute_tvdi(
        image.select('NDVI_90'),
        image.select('LST_Day_1km_90'),
        'NDVI_90',
        'LST_Day_1km_90'
    )

    return image.addBands([tvdi30, tvdi60, tvdi90])

# Function to create 7x7x2 image representation for locust reports


def create_locust_report_image(point, date, spatial_resolution):
    """Create a 7x7x2 image representation for locust reports"""
    geometry = point.geometry()
    center_lat = geometry.coordinates().get(1)
    center_lon = geometry.coordinates().get(0)

    # Create a 7x7 grid centered on the current location
    grid_size = 7
    half_grid_size = math.floor(grid_size / 2)
    grid_cells = []

    # Calculate the step size in degrees based on the spatial resolution
    # Convert meters to approximate degrees
    step_size = spatial_resolution / 111000

    # Create a feature collection of grid cells
    for i in range(-half_grid_size, half_grid_size + 1):
        for j in range(-half_grid_size, half_grid_size + 1):
            cell_lon = ee.Number(center_lon).add(
                ee.Number(j).multiply(step_size))
            cell_lat = ee.Number(center_lat).add(
                ee.Number(i).multiply(step_size))

            cell = ee.Feature(
                ee.Geometry.Rectangle([
                    cell_lon.subtract(
                        step_size/2), cell_lat.subtract(step_size/2),
                    cell_lon.add(step_size/2), cell_lat.add(step_size/2)
                ]),
                {
                    'row': i + half_grid_size,
                    'col': j + half_grid_size,
                    'center_lon': cell_lon,
                    'center_lat': cell_lat
                }
            )

            grid_cells.append(cell)

    grid = ee.FeatureCollection(grid_cells)

    # Get previous dates for time-series analysis
    prev_date30 = date.advance(-30, 'days')
    prev_date60 = date.advance(-60, 'days')
    prev_date90 = date.advance(-90, 'days')

    # Filter locust reports for each time period
    locust_reports = ee.FeatureCollection(fao_report_asset_id)

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

    image30 = create_period_image(counts30, '30')
    image60 = create_period_image(counts60, '60')
    image90 = create_period_image(counts90, '90')

    return ee.Image.cat([image30, image60, image90])

# Function to check if an image has missing bands/data


def check_image_for_missing_data(image, geometry):
    """Check if an image has any missing data in the specified bands"""
    try:
        # Get a sample value for each band to check if data exists
        sample = image.sample(
            region=geometry,
            scale=1000,
            numPixels=1,
            seed=42,
            dropNulls=False
        ).first()

        # Convert to dictionary to check for nulls
        sample_dict = sample.toDictionary().getInfo()

        # Check if any values are None or NaN
        for key, value in sample_dict.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                logging.warning(f"Band {key} has null or NaN values")
                return True

        return False
    except Exception as e:
        logging.error(f"Error checking for missing data: {str(e)}")
        return True  # Assume data is missing if there's an error

# Function to extract time-lagged environmental data


def extract_time_lagged_data(point):
    """Extract time-lagged environmental data for a point"""
    # Use the parsed date we stored earlier
    date = ee.Date(point.get("parsed_date")) if point.get(
        "parsed_date") else ee.Date(datetime.datetime.now().strftime("%Y-%m-%d"))

    geometry = point.geometry()

    lags = {
        "30": date.advance(-30, "days"),
        "60": date.advance(-60, "days"),
        "90": date.advance(-90, "days")
    }

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
        return None

    # If we have all variables, continue with image creation
    images = list(variables.values()) + [
        elevation,
        landcover,
        sand_content0_5,
        sand_content5_15,
        sand_content15_30
    ]

    combined_image = ee.Image.cat(images)
    combined_image = calculate_tci(combined_image)
    combined_image = calculate_vhi(combined_image)

    # Create 7x7x2 locust report image
    locust_report_image = create_locust_report_image(
        point, date, 1000)  # 1km resolution

    # Combine all data
    final_image = ee.Image.cat([combined_image, locust_report_image])

    # Check for missing data in the final image
    if check_image_for_missing_data(final_image, geometry.buffer(10000)):
        logging.warning("Final image has missing data")
        return None

    return final_image

# Function to create an export task


def create_export_task(feature_index, feature):
    """Create an export task for a feature"""
    try:
        # Ensure feature is an ee.Feature object
        # if not isinstance(feature, ee.Feature):
        #     logging.warning(
        #         f"Input for feature_index {feature_index} is not an ee.Feature. Skipping.")
        #     return None

        # Get the observation date client-side
        try:
            # Server-side check if 'Obs Date' exists and is not null
            has_obs_date = feature.propertyNames().contains('Obs Date')
            obs_date_is_null = ee.Algorithms.IsEqual(
                feature.get('Obs Date'), None)

            # Use ee.Algorithms.If for conditional check server-side
            should_process_date = ee.Algorithms.If(
                has_obs_date.And(obs_date_is_null.Not()),
                True,
                False
            )

            if not should_process_date.getInfo():
                logging.warning(
                    f"Feature {feature_index} has missing or null 'Obs Date'. Skipping.")
                return None

            # Get the raw date value safely
            obs_date_raw = feature.get('Obs Date')
            obs_date_client = obs_date_raw.getInfo()  # GetInfo only if needed

            # Validate client-side date format
            if not isinstance(obs_date_client, str) or '/' not in obs_date_client:
                logging.warning(
                    f"Feature {feature_index} has invalid date format: {obs_date_client}. Skipping.")
                return None

            # Process the date parts carefully
            date_parts = obs_date_client.split(' ')[0].split('/')
            if len(date_parts) < 3:
                logging.warning(
                    f"Feature {feature_index} invalid date parts: {date_parts}. Skipping.")
                return None

            month, day, year = date_parts[0].zfill(
                2), date_parts[1].zfill(2), date_parts[2]

            # Validate year
            if len(year) != 4 or not (1900 <= int(year) <= datetime.datetime.now().year + 1):
                logging.warning(
                    f"Feature {feature_index} has invalid year: {year}. Skipping.")
                return None

            formatted_date = f"{year}-{month}-{day}"
            ee_date = ee.Date(formatted_date)  # Create EE Date server-side
            logging.info(
                f"Feature {feature_index}: Successfully parsed date: {formatted_date}")

        except Exception as e:
            logging.warning(
                f"Error parsing date for feature {feature_index}: {e}. Raw date: {obs_date_client if 'obs_date_client' in locals() else 'N/A'}. Skipping.")
            return None

        # Get the locust presence value server-side
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

        except Exception as e:
            logging.warning(
                f"Error getting presence for feature {feature_index}: {e}. Skipping.")
            return None

        # Set the parsed date on the feature
        feature_with_parsed_date = feature.set('parsed_date', ee_date)

        # Extract time-lagged data
        time_lagged_data = extract_time_lagged_data(feature_with_parsed_date)

        # Skip if data is missing
        if time_lagged_data is None:
            logging.warning(
                f"Missing environmental data for feature {feature_index}. Skipping task creation.")
            return None

        # Prepare for export
        time_lagged_data = time_lagged_data.toFloat()
        # Ensure geometry exists before buffering
        feature_geometry = feature.geometry()
        if feature_geometry is None:
            logging.warning(
                f"Feature {feature_index} has no geometry. Skipping.")
            return None
        patch_geometry = feature_geometry.buffer(10000)  # 10km buffer

        # Create multi-band image with label
        multi_band_image = ee.Image.cat([
            time_lagged_data,
            ee.Image.constant(
                1 if presence == 'PRESENT' else 0).toFloat().rename('label')
        ]).clip(patch_geometry)

        # Create export task
        export_description = f'locust_{formatted_date}_label_{1 if presence == "PRESENT" else 0}_{feature_index + 1}'

        export_task = ee.batch.Export.image.toDrive(
            image=multi_band_image,
            description=export_description,
            scale=common_scale,
            region=patch_geometry,
            maxPixels=1e13,
            crs=common_projection,
            folder='Locust_Export'
        )
        logging.info(
            f"Created export task for feature {feature_index}: {export_description}")
        return export_task, export_description

    except ee.EEException as e:
        logging.error(
            f"Earth Engine error creating export task for feature {feature_index}: {str(e)}. Skipping.")
        return None
    except Exception as e:
        logging.error(
            f"General error creating export task for feature {feature_index}: {str(e)}. Skipping.")
        return None

# Class to manage task queue


class TaskManager:
    def __init__(self, max_concurrent=240, max_retries=3, retry_delay=300):
        self.task_queue = Queue()
        self.active_tasks = {}  # task_id -> (task, description, attempts)
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.lock = Lock()
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.total_count = 0
        self.running = True
        self.last_active_count = 0  # Track last active count

        # Start worker threads
        self.task_monitor = Thread(target=self._monitor_tasks)
        self.task_monitor.daemon = True
        self.task_monitor.start()

    def add_task(self, task_tuple):
        """Add a task to the queue"""
        if task_tuple is None:
            with self.lock:
                self.skipped_count += 1
                self.total_count += 1
            return

        task, description = task_tuple
        with self.lock:
            self.total_count += 1

        # (task, description, attempts)
        self.task_queue.put((task, description, 0))
        logging.info(f"Added task to queue: {description}")

    def _start_task(self, task, description, attempts):
        """Start a task and add it to active tasks"""
        task_id = f"{description}_{attempts}"
        with self.lock:
            if task_id in self.active_tasks:
                logging.warning(f"Task {task_id} already in active tasks!")
                return

            self.active_tasks[task_id] = (task, description, attempts)

        try:
            task.start()
            logging.info(f"Started task: {description} (attempt {attempts+1})")
        except Exception as e:
            logging.error(f"Error starting task {description}: {str(e)}")
            with self.lock:
                self.failed_count += 1
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

    def _monitor_tasks(self):
        """Monitor running tasks and start new ones as needed"""
        while self.running:
            try:
                # Process any completed tasks
                task_ids_to_remove = []

                with self.lock:
                    active_task_ids = list(self.active_tasks.keys())

                for task_id in active_task_ids:
                    with self.lock:
                        if task_id not in self.active_tasks:
                            continue
                        task, description, attempts = self.active_tasks[task_id]

                    try:
                        status = task.status()
                        state = status['state']

                        if state == 'COMPLETED':
                            logging.info(f"Task completed: {description}")
                            with self.lock:
                                self.completed_count += 1
                                task_ids_to_remove.append(task_id)
                        elif state == 'FAILED':
                            logging.warning(
                                f"Task failed: {description} - {status.get('error_message', 'Unknown error')}")
                            with self.lock:
                                task_ids_to_remove.append(task_id)

                            # Add to retry queue if not exceeded max retries
                            if attempts < self.max_retries:
                                logging.info(
                                    f"Scheduling retry for task: {description} (attempt {attempts+1})")
                                # Wait before retrying
                                time.sleep(self.retry_delay)
                                self.task_queue.put(
                                    (task, description, attempts + 1))
                            else:
                                logging.error(
                                    f"Task failed after {self.max_retries} attempts: {description}")
                                with self.lock:
                                    self.failed_count += 1
                        elif state in ['CANCELLED', 'CANCEL_REQUESTED']:
                            logging.warning(f"Task cancelled: {description}")
                            with self.lock:
                                self.failed_count += 1
                                task_ids_to_remove.append(task_id)
                    except Exception as e:
                        logging.error(
                            f"Error monitoring task {description}: {str(e)}")
                        with self.lock:
                            task_ids_to_remove.append(task_id)

                # Remove completed/failed tasks
                with self.lock:
                    for task_id in task_ids_to_remove:
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]

                # Start new tasks if there's room
                with self.lock:
                    active_count = len(self.active_tasks)

                while active_count < self.max_concurrent and not self.task_queue.empty():
                    try:
                        task, description, attempts = self.task_queue.get(
                            block=False)
                        self._start_task(task, description, attempts)
                        active_count += 1
                    except Exception as e:
                        if not isinstance(e, Exception):
                            logging.error(f"Error starting new task: {str(e)}")
                        break

                # Print status only when active_count increases by one
                with self.lock:
                    active_count = len(self.active_tasks)
                    if active_count > self.last_active_count:
                        logging.info(f"Status: {active_count} active, {self.task_queue.qsize()} queued, "
                                     f"{self.completed_count} completed, {self.failed_count} failed, {self.skipped_count} skipped, "
                                     f"{self.total_count} total")
                        self.last_active_count = active_count

                # Sleep to avoid busy waiting
                time.sleep(60)

            except Exception as e:
                logging.error(f"Error in task monitor: {str(e)}")
                time.sleep(10)  # Sleep a bit before retrying

    def wait_until_complete(self):
        """Wait until all tasks are completed"""
        while self.running and (not self.task_queue.empty() or len(self.active_tasks) > 0):
            time.sleep(10)

        return self.completed_count, self.failed_count, self.skipped_count

    def shutdown(self):
        """Shutdown the task manager"""
        self.running = False
        if self.task_monitor.is_alive():
            self.task_monitor.join(timeout=60)

# Function to save progress to a file


def save_progress(progress_file, processed_indices, completed_count, failed_count, skipped_count):
    """Save progress to a JSON file"""
    try:
        progress_data = {
            'processed_indices': processed_indices,
            'completed_count': completed_count,
            'failed_count': failed_count,
            'skipped_count': skipped_count,
            'timestamp': datetime.datetime.now().isoformat()
        }

        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)

        logging.info(f"Progress saved to {progress_file}")

    except Exception as e:
        logging.error(f"Error saving progress: {str(e)}")

# Function to load progress from a file


def load_progress(progress_file):
    """Load progress from a JSON file"""
    try:
        if not os.path.exists(progress_file):
            logging.info(f"No progress file found at {progress_file}")
            return set(), 0, 0, 0

        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        processed_indices = set(progress_data.get('processed_indices', []))
        completed_count = progress_data.get('completed_count', 0)
        failed_count = progress_data.get('failed_count', 0)
        skipped_count = progress_data.get('skipped_count', 0)

        logging.info(f"Loaded progress from {progress_file}: {len(processed_indices)} processed, "
                     f"{completed_count} completed, {failed_count} failed, {skipped_count} skipped")

        return processed_indices, completed_count, failed_count, skipped_count

    except Exception as e:
        logging.error(f"Error loading progress: {str(e)}")
        return set(), 0, 0, 0


def process_feature_batch(features, start_idx, end_idx, task_manager, processed_indices, progress_file):
    """Process a batch of features"""
    batch_size = end_idx - start_idx
    batch_indices = list(range(start_idx, end_idx))

    logging.info(f"Processing batch from {start_idx} to {end_idx-1}")

    # Create export tasks for each feature
    for i, idx in enumerate(batch_indices):
        if idx in processed_indices:
            logging.info(f"Skipping already processed feature {idx}")
            continue

        try:
            feature = ee.Feature(features.get(idx))
            task_tuple = create_export_task(idx, feature)
            task_manager.add_task(task_tuple)
            processed_indices.add(idx)

            # Save progress after every 10 features
            if i % 10 == 0:
                save_progress(
                    progress_file,
                    list(processed_indices),
                    task_manager.completed_count,
                    task_manager.failed_count,
                    task_manager.skipped_count
                )

        except Exception as e:
            logging.error(f"Error processing feature {idx}: {str(e)}")

    # Save progress at the end of the batch
    save_progress(
        progress_file,
        list(processed_indices),
        task_manager.completed_count,
        task_manager.failed_count,
        task_manager.skipped_count
    )

    logging.info(f"Batch {start_idx}-{end_idx-1} submitted to queue")


def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Export locust data from Earth Engine')
    parser.add_argument('--test', action='store_true',
                        help='Run with a single test point')
    parser.add_argument('--batch-size', type=int, default=250,
                        help='Number of features to process in one batch')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Index to start processing from')
    parser.add_argument('--max-features', type=int, default=None,
                        help='Maximum number of features to process')
    parser.add_argument('--presence-only', action='store_true',
                        help='Process only presence points')
    parser.add_argument('--absence-only', action='store_true',
                        help='Process only absence points')
    parser.add_argument('--progress-file', type=str, default='locust_export_progress.json',
                        help='File to save/load progress')
    parser.add_argument('--log-file', type=str,
                        default='locust_export.log', help='Log file name')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_file)

    # Initialize Earth Engine
    initialize_ee()

    # Define global variables
    global et_boundary, common_scale, common_projection, fao_report_asset_id
    et_boundary = get_ethiopia_boundary()
    common_scale = 250
    common_projection = 'EPSG:4326'
    fao_report_asset_id = 'projects/desert-locust-forcast/assets/FAO_filtered_data_2000'

    logging.info("Loading FAO locust data...")

    # Load FAO locust data
    locust_data = ee.FeatureCollection(fao_report_asset_id)

    # Filter out features with null 'x' or 'y'
    locust_data = locust_data.filter(ee.Filter.And(
        ee.Filter.neq('Longitude', None),
        ee.Filter.neq('Latitude', None)
    ))

    # Filter out features with null 'Obs Date'
    locust_data = locust_data.filter(ee.Filter.neq('Obs Date', None))

    # Filter out features with null 'Locust Presence'
    locust_data = locust_data.filter(ee.Filter.neq('Locust Presence', None))

    # Log the impact of filtering to diagnose data quality
    original_count = ee.FeatureCollection(fao_report_asset_id).size().getInfo()
    filtered_count = locust_data.size().getInfo()
    logging.info(
        f"Original features: {original_count}, after filtering null x/y: {filtered_count}")

    # Filter for presence and absence points
    if args.presence_only:
        logging.info("Processing only presence points")
        filtered_data = locust_data.filter(
            ee.Filter.eq('Locust Presence', 'PRESENT'))
    elif args.absence_only:
        logging.info("Processing only absence points")
        filtered_data = locust_data.filter(
            ee.Filter.eq('Locust Presence', 'ABSENT'))
    else:
        filtered_data = locust_data

    # Get count of features
    feature_count = filtered_data.size().getInfo()
    logging.info(f'Total features to process: {feature_count}')

    # Sorted by Year
    filtered_data = filtered_data.sort('Year', ascending=False)

    logging.info(
        f'Indexed collection: {filtered_data.first().getInfo()}')

    # Limit to max features if specified
    if args.max_features is not None:
        max_idx = min(args.start_index + args.max_features, feature_count)
        logging.info(
            f"Processing features from index {args.start_index} to {max_idx-1}")
    else:
        max_idx = feature_count
        logging.info(
            f"Processing all features from index {args.start_index} to {feature_count-1}")

    # Load progress if available
    processed_indices, completed_count, failed_count, skipped_count = load_progress(
        args.progress_file)

    # Test mode - process a single point
    if args.test:
        single_point = filtered_data.filter(
            ee.Filter.eq('index', args.start_index)).first()
        logging.info('Processing test point...')
        logging.info(
            f'Feature Property Names: {single_point.propertyNames().getInfo()}')
        task_tuple = create_export_task(
            args.start_index, single_point.getInfo())
        if task_tuple:
            task, description = task_tuple
            task.start()
            logging.info(f'Test export task started: {description}')

            # Wait for the task to complete or fail
            while task.status()['state'] in ('READY', 'RUNNING'):
                logging.info(f"Test task status: {task.status()['state']}")
                time.sleep(10)

            logging.info(
                f"Test task completed with state: {task.status()['state']}")

            if task.status()['state'] == 'COMPLETED':
                logging.info("Test export successful.")
            else:
                logging.error(
                    f"Test export failed: {task.status().get('error_message', 'Unknown error')}")
        else:
            logging.warning(
                "Test export task could not be created, possibly due to missing data.")

        return

    # Initialize task manager for batch processing
    task_manager = TaskManager(
        max_concurrent=240, max_retries=3, retry_delay=300)

    try:
        # Process in batches to avoid memory issues
        batch_size = args.batch_size
        start_idx = args.start_index

        for batch_start in range(start_idx, max_idx, batch_size):
            batch_end = min(batch_start + batch_size, max_idx)
            process_feature_batch(
                features,
                batch_start,
                batch_end,
                task_manager,
                processed_indices,
                args.progress_file
            )

        # Wait for all tasks to complete
        logging.info("All batches submitted. Waiting for tasks to complete...")
        completed, failed, skipped = task_manager.wait_until_complete()

        logging.info(
            f"All tasks completed: {completed} successful, {failed} failed, {skipped} skipped")

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving progress...")
        save_progress(
            args.progress_file,
            list(processed_indices),
            task_manager.completed_count,
            task_manager.failed_count,
            task_manager.skipped_count
        )
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
    finally:
        # Shutdown task manager
        task_manager.shutdown()

        # Save final progress
        save_progress(
            args.progress_file,
            list(processed_indices),
            task_manager.completed_count,
            task_manager.failed_count,
            task_manager.skipped_count
        )

        logging.info("Script completed. Final progress saved.")


if __name__ == "__main__":
    main()
