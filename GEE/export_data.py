import ee
import time
from datetime import datetime
import logging
from typing import List, Dict
import math

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


def get_environment_data(report: ee.Feature, aoi: ee.Geometry) -> ee.Image:
    """Get environmental data for a specific report timestamp"""
    report_date = ee.Date(report.get('FINISHDATE'))
    start = report_date.advance(-7, 'day')
    end = report_date.advance(7, 'day')

    # Process HLS (Harmonized Landsat Sentinel) data
    hls_image = ee.ImageCollection("NASA/HLS/HLSL30/v002") \
        .filterBounds(aoi) \
        .filterDate(start, end)

    def process_hls(image):
        ndvi = image.normalizedDifference(["B5", "B4"]).rename("NDVI")
        evi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                'NIR': image.select("B5"),
                'RED': image.select("B4"),
                'BLUE': image.select("B2")
            }
        ).rename("EVI")
        return image.addBands(ndvi).addBands(evi)

    hls_image = hls_image.map(process_hls)
    hls_final = ee.Image(ee.Algorithms.If(
        hls_image.size().gt(0),
        hls_image.select(["NDVI", "EVI"]).median(),
        hls_image.first()
    )).clip(aoi)

    # Get other environmental data
    soil_moisture = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007") \
        .filterBounds(aoi).filterDate(start, end) \
        .select("sm_surface").median().clip(aoi)

    lst = ee.ImageCollection("MODIS/061/MOD11A2") \
        .filterBounds(aoi).filterDate(start, end) \
        .select("LST_Day_1km") \
        .map(lambda img: img.multiply(0.02).subtract(273.15)) \
        .median().clip(aoi)

    # Process ERA5-Land data for humidity and wind
    era5_hourly = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
        .filterBounds(aoi).filterDate(start, end)

    humidity = era5_hourly.select(["dewpoint_temperature_2m", "temperature_2m"]) \
        .map(lambda img: ee.Image(100).multiply(
            ee.Image(112).subtract(
                img.select("temperature_2m").subtract(273.15)
                .subtract(img.select("dewpoint_temperature_2m").subtract(273.15))
                .multiply(5)
            )
        ).exp().divide(100)).median().clip(aoi)

    wind_u = era5_hourly.select("u_component_of_wind_10m").median().clip(aoi)
    wind_v = era5_hourly.select("v_component_of_wind_10m").median().clip(aoi)

    # Get precipitation data
    precipitation = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(aoi).filterDate(start, end) \
        .select("precipitation").sum().clip(aoi)

    # Get elevation data
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)

    # Combine all bands
    return ee.Image.cat([
        hls_final,
        soil_moisture,
        lst,
        humidity,
        wind_u,
        wind_v,
        precipitation,
        elevation
    ])


def create_export_task(report: ee.Feature, index: int, aoi: ee.Geometry) -> ee.batch.Task:
    """Create an export task for a single report"""
    report_image = get_environment_data(report, aoi).toFloat()
    finish_time = ee.Date(report.get("FINISHDATE")
                          ).format("YYYY-MM-dd").getInfo()

    task = ee.batch.Export.image.toDrive(
        image=report_image,
        description=f'locust_data_{index}_{finish_time}',
        scale=1000,  # 1km resolution
        region=aoi,
        maxPixels=1e13,
        crs="EPSG:4326",
        folder='Thesis'
    )
    return task


def monitor_and_export(locust_points: ee.FeatureCollection, batch_size: int = 250):
    """Monitor and manage export tasks with queue limitation"""
    aoi = get_ethiopia_boundary()
    total_reports = locust_points.size().getInfo()
    num_batches = math.ceil(total_reports / batch_size)
    active_tasks: Dict[str, ee.batch.Task] = {}
    completed_count = 0

    logging.info(
        f"Starting export of {total_reports} reports in {num_batches} batches")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        batch_reports = locust_points.toList(batch_size, start_idx)

        # Get report list for this batch
        reports = batch_reports.getInfo()

        for idx, report in enumerate(reports):
            global_idx = start_idx + idx

            # Wait if we have too many active tasks
            while len(active_tasks) >= 250:  # Keep buffer of 50 for safety
                time.sleep(60)  # Check every minute
                active_tasks = {task_id: task for task_id, task in active_tasks.items()
                                if task.status()['state'] in ('READY', 'RUNNING')}

            # Create and start new task
            report_feature = ee.Feature(report)
            task = create_export_task(report_feature, global_idx, aoi)
            task.start()

            task_id = task.status()['id']
            active_tasks[task_id] = task

            logging.info(f"Started export {global_idx + 1}/{total_reports}")

            # Update completion count
            completed_count += 1
            if completed_count % 100 == 0:
                logging.info(
                    f"Progress: {completed_count}/{total_reports} exports initiated")

            time.sleep(2)  # Small delay between task creation

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

        # Load FAO locust reports (you'll need to define how to load this)
        locust_points = ee.FeatureCollection(
            'projects/desert-locust-forcast/assets/FAO_Swarm_Report_RAW_2019_2021')

        monitor_and_export(locust_points)

    except Exception as e:
        logging.error(f"Export process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
