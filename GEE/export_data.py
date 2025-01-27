import ee
import time
import logging
from datetime import datetime
import math
from typing import Dict

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def initialize_ee():
    """Initialize Earth Engine"""
    try:
        ee.Initialize()
        logging.info("Earth Engine initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Earth Engine: {str(e)}")
        raise


def get_ethiopia_boundary():
    """Get Ethiopia boundary from GAUL dataset"""
    gaul_dataset = ee.FeatureCollection("FAO/GAUL/2015/level1")
    et_boundary = gaul_dataset.filter(ee.Filter.eq('ADM0_NAME', 'Ethiopia'))
    return et_boundary.geometry()


def calculate_vhi(image):
    """Calculate Vegetation Health Index"""
    ndvi30 = image.select('NDVI_30')
    ndvi60 = image.select('NDVI_60')
    ndvi90 = image.select('NDVI_90')

    tci30 = image.select('TCI_30')
    tci60 = image.select('TCI_60')
    tci90 = image.select('TCI_90')

    vhi30 = ndvi30.multiply(0.5).add(tci30.multiply(0.5)).rename('VHI_30')
    vhi60 = ndvi60.multiply(0.5).add(tci60.multiply(0.5)).rename('VHI_60')
    vhi90 = ndvi90.multiply(0.5).add(tci90.multiply(0.5)).rename('VHI_90')

    return image.addBands([vhi30, vhi60, vhi90])


def calculate_tci(image):
    """Calculate Temperature Condition Index"""
    lst30 = image.select('LST_30')
    lst60 = image.select('LST_60')
    lst90 = image.select('LST_90')

    tci30 = lst30.subtract(273.15).multiply(0.1).rename('TCI_30')
    tci60 = lst60.subtract(273.15).multiply(0.1).rename('TCI_60')
    tci90 = lst90.subtract(273.15).multiply(0.1).rename('TCI_90')

    return image.addBands([tci30, tci60, tci90])


def extract_time_lagged_data(point):
    """Extract time-lagged environmental variables for a point"""
    date = ee.Date(point.get('FINISHDATE'))
    point_geometry = point.geometry()

    # Define time lags
    lag30 = date.advance(-30, 'days')
    lag60 = date.advance(-60, 'days')
    lag90 = date.advance(-90, 'days')

    # Extract all variables with time lags
    collections = {
        'NDVI': {
            'collection': 'MODIS/061/MOD13A2',
            'bands': ['NDVI', 'EVI'],
            'reducer': 'mean'
        },
        'LST': {
            'collection': 'MODIS/061/MOD11A2',
            'bands': ['LST_Day_1km'],
            'reducer': 'mean'
        },
        'CHIRPS': {
            'collection': 'UCSB-CHG/CHIRPS/DAILY',
            'bands': ['precipitation'],
            'reducer': 'sum'
        },
        'ERA5': {
            'collection': 'ECMWF/ERA5/DAILY',
            'bands': ['u_component_of_wind_10m', 'v_component_of_wind_10m'],
            'reducer': 'mean'
        },
        'SMAP': {
            'collection': 'NASA/SMAP/SPL4SMGP/007',
            'bands': ['sm_surface'],
            'reducer': 'mean'
        }
    }

    all_bands = []

    for var_name, config in collections.items():
        for lag, lag_days in [('30', lag30), ('60', lag60), ('90', lag90)]:
            collection = ee.ImageCollection(config['collection']) \
                .filterBounds(point_geometry) \
                .filterDate(lag_days, date)

            for band in config['bands']:
                img = collection.select(band)
                if config['reducer'] == 'mean':
                    img = img.mean()
                else:
                    img = img.sum()

                new_name = f"{band}_{lag}"
                all_bands.append(img.rename(new_name))

    # Combine all bands
    combined_image = ee.Image.cat(all_bands)

    # Calculate indices
    return calculate_vhi(calculate_tci(combined_image))


def aggregate_over_buffer(image, point, buffer_radius):
    """Aggregate environmental data over a buffer zone"""
    point_geometry = point.geometry().buffer(buffer_radius)
    return image.reduceRegion({
        'reducer': ee.Reducer.mean(),
        'geometry': point_geometry,
        'scale': 1000,
        'maxPixels': 1e13
    })


def create_export_task(feature_index: int, feature: ee.Feature, aoi: ee.Geometry) -> ee.batch.Task:
    """Create an export task for a single feature"""
    finish_date = ee.Date(feature.get('FINISHDATE')
                          ).format('YYYY-MM-dd').getInfo()

    # Extract time-lagged data
    time_lagged_data = extract_time_lagged_data(feature).toFloat()

    # Create multi-band image
    multi_band_image = ee.Image.cat([
        time_lagged_data,
        ee.Image.constant(feature.get('LOCPRESENT')).toFloat().rename('label')
    ]).clip(aoi)

    # Create export task
    task = ee.batch.Export.image.toDrive({
        'image': multi_band_image,
        'description': f'locust_feature_{feature_index}_{finish_date}',
        'scale': 1000,
        'region': aoi,
        'maxPixels': 1e13,
        'crs': 'EPSG:4326',
        'folder': 'Thesis'
    })

    return task


def monitor_and_export(fao_report_asset_id: str, batch_size: int = 250):
    """Monitor and manage export tasks with queue limitation"""
    aoi = get_ethiopia_boundary()

    # Load FAO desert locust report
    locust_points = ee.FeatureCollection(fao_report_asset_id)

    # Get total number of features
    total_features = locust_points.size().getInfo()
    num_batches = math.ceil(total_features / batch_size)

    logging.info(
        f"Starting export of {total_features} features in {num_batches} batches")

    active_tasks: Dict[str, ee.batch.Task] = {}
    completed_count = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        batch_features = locust_points.toList(batch_size, start_idx)

        # Get feature list for this batch
        features = batch_features.getInfo()

        for idx, feature in enumerate(features):
            global_idx = start_idx + idx

            # Wait if we have too many active tasks
            while len(active_tasks) >= 250:
                time.sleep(60)
                active_tasks = {task_id: task for task_id, task in active_tasks.items()
                                if task.status()['state'] in ('READY', 'RUNNING')}

            # Create and start new task
            feature_ee = ee.Feature(feature)
            task = create_export_task(global_idx, feature_ee, aoi)
            task.start()

            task_id = task.status()['id']
            active_tasks[task_id] = task

            logging.info(f"Started export {global_idx + 1}/{total_features}")

            completed_count += 1
            if completed_count % 100 == 0:
                logging.info(
                    f"Progress: {completed_count}/{total_features} exports initiated")

            time.sleep(2)

    # Wait for remaining tasks to complete
    while active_tasks:
        time.sleep(60)
        active_tasks = {task_id: task for task_id, task in active_tasks.items()
                        if task.status()['state'] in ('READY', 'RUNNING')}
        logging.info(
            f"Waiting for {len(active_tasks)} remaining tasks to complete")

    logging.info("All exports completed successfully")


def main():
    """Main function to run the export process"""
    try:
        initialize_ee()

        # Replace with your FAO report asset ID
        fao_report_asset_id = 'projects/desert-locust-forcast/assets/FAO_Swarm_Report_RAW_2019_2021'

        monitor_and_export(fao_report_asset_id)

    except Exception as e:
        logging.error(f"Export process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
