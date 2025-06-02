import ee
import time
import json
import os
import logging
import psutil
import gc
from datetime import datetime
from pathlib import Path

ee.Initialize()

# Set up logging
LOG_FILE = 'export_tasks.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Set up progress tracking
PROGRESS_FILE = 'export_progress.json'

# Stats tracking
stats = {
    'total_tasks_started': 0,
    'tasks_started_this_session': 0,
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'peak_memory_usage_mb': 0
}

# Function to get current memory usage


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # Convert bytes to MB
    return memory_mb

# Function to log memory usage and update peak if needed


def log_memory_usage(context=""):
    current_memory = get_memory_usage()
    if current_memory > stats['peak_memory_usage_mb']:
        stats['peak_memory_usage_mb'] = current_memory

    message = f"Memory usage: {current_memory:.2f} MB"
    if context:
        message = f"{context} - {message}"

    logging.info(message)
    return current_memory


# Log initial memory usage
log_memory_usage("Initial")

# Load progress from file if exists


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress_data = json.load(f)
            # If the file has stats, use them
            if 'stats' in progress_data:
                stats.update(progress_data['stats'])
                # Reset session counter and memory peak
                stats['tasks_started_this_session'] = 0
                stats['peak_memory_usage_mb'] = get_memory_usage()
            return progress_data
    return {'exported_features': [], 'stats': stats}

# Save progress to file


def save_progress(progress):
    # Update stats in progress data
    progress['stats'] = stats
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


# Initialize or load progress
progress = load_progress()
exported_features = set(progress['exported_features'])
logging.info(f"Loaded {len(exported_features)} previously exported features")
logging.info(
    f"Total tasks started before this session: {stats['total_tasks_started']}")

# Load the feature collection
fc = ee.FeatureCollection(
    'projects/desert-locust-forcast/assets/FAO_archival_data_extracted_2019_Ethiopia')

# Filter for Ethiopia
gaul = ee.FeatureCollection('FAO/GAUL/2015/level0')
ethiopia = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Ethiopia'))
fc = fc.filterBounds(ethiopia)

# Parse date and add as string property


def parse_date(feature):
    obs_date_str = ee.String(feature.get('formatted_date'))
    return feature.set('obs_date_str', obs_date_str)


fc = fc.map(parse_date)

# Filter by date
fc = fc.filter(ee.Filter.And(
    ee.Filter.gte('obs_date_str', '2019-01-01'),
    ee.Filter.lte('obs_date_str', '2023-12-31')
))

# Function to start export


def start_export(feature_dict):
    feature_id = feature_dict['id']

    # Skip if already exported
    if feature_id in exported_features:
        logging.debug(f"Skipping already exported feature: {feature_id}")
        return

    try:
        obs_date_str = feature_dict['properties']['obs_date_str']
        obs_date = ee.Date(obs_date_str)
        geometry = ee.Geometry(feature_dict['geometry'])
        region = geometry.buffer(1120).bounds()
        start_date = obs_date.advance(-15, 'day')
        end_date = obs_date.advance(15, 'day')
        s2 = ee.ImageCollection('COPERNICUS/S2_SR')
        bands = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
        s2_filtered = s2.filterDate(start_date, end_date).filterBounds(geometry).filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('CLOUDY_PIXEL_PERCENTAGE').first()
        if s2_filtered:
            image = s2_filtered.select(bands).clip(region)
            index = feature_dict['properties']['OBJECTID']
            presence = feature_dict['properties']['Locust Presence']
            label = '1' if presence == 'PRESENT' else '0'
            year, month, day = obs_date_str.split('-')
            print_date = f"{year}_{month}_{day}"
            description = f"patch_idx_{index}_{print_date}_label_{label}"
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder='desert_locust_patches_ethiopia',
                fileNamePrefix=description,
                region=region,
                scale=10,
                crs='EPSG:4326'
            )
            task.start()

            # Update task counters
            stats['total_tasks_started'] += 1
            stats['tasks_started_this_session'] += 1

            # Log memory usage after starting a task
            mem_usage = log_memory_usage(f"After starting task {index}")

            logging.info(f"Started task: {description} (ID: {feature_id})")
            logging.info(
                f"Tasks started: {stats['tasks_started_this_session']} this session, {stats['total_tasks_started']} total")

            # Mark as exported and save progress
            exported_features.add(feature_id)
            progress['exported_features'] = list(exported_features)
            save_progress(progress)

            # Force garbage collection if memory usage is high (over 80% of peak)
            if mem_usage > stats['peak_memory_usage_mb'] * 0.8:
                gc.collect()
                logging.info(
                    f"Performed garbage collection. Memory after GC: {get_memory_usage():.2f} MB")
        else:
            logging.warning(f"No image found for feature: {feature_id}")
    except Exception as e:
        logging.error(f"Error processing feature {feature_id}: {e}")


# Get size
size = fc.size().getInfo()
batch_size = 100

logging.info(f"Processing {size} features in batches of {batch_size}...")
logging.info(f"Session started at: {stats['start_time']}")

for offset in range(0, size, batch_size):
    logging.info(f"Processing batch starting at offset {offset}")
    features_list = fc.toList(batch_size, offset).getInfo()

    for feature_dict in features_list:
        start_export(feature_dict)

    # Log memory usage after each batch
    log_memory_usage(f"After batch at offset {offset}")

    # Save progress after each batch
    progress['exported_features'] = list(exported_features)
    save_progress(progress)
    logging.info(f"Completed batch. Total exported: {len(exported_features)}")

    # Force garbage collection after each batch
    gc.collect()
    logging.info(
        f"Performed garbage collection after batch. Memory: {get_memory_usage():.2f} MB")

    time.sleep(1)  # Optional delay to avoid overwhelming the system

# Calculate session duration
end_time = datetime.now()
start_time = datetime.strptime(stats['start_time'], '%Y-%m-%d %H:%M:%S')
duration = end_time - start_time
duration_str = str(duration).split('.')[0]  # Remove microseconds

# Final memory usage logging
log_memory_usage("Final")

logging.info(
    f"Export process completed. Total exported features: {len(exported_features)}")
logging.info(
    f"Tasks started in this session: {stats['tasks_started_this_session']}")
logging.info(f"Total tasks started overall: {stats['total_tasks_started']}")
logging.info(f"Peak memory usage: {stats['peak_memory_usage_mb']:.2f} MB")
logging.info(f"Session duration: {duration_str}")
