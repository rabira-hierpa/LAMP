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


class GEEExporter:
    def __init__(self):
        self.common_scale = 1000  # 1 km resolution
        self.common_projection = 'EPSG:4326'  # WGS84 projection

    def initialize_ee(self):
        """Initialize Earth Engine"""
        try:
            ee.Initialize()
            logging.info("Earth Engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Earth Engine: {str(e)}")
            raise

    def get_ethiopia_boundary(self) -> ee.Geometry:
        """Get Ethiopia boundary from the simplified LSIB dataset."""
        lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        et_boundary = lsib.filter(ee.Filter.eq("country_na", "Ethiopia"))
        return et_boundary.geometry().simplify(maxError=1000)  # Simplify geometry

    def calculate_vhi(self, image):
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

    def calculate_tci(self, image):
        """Calculate Temperature Condition Index"""
        lst30 = image.select('LST_Day_1km_30')
        lst60 = image.select('LST_Day_1km_60')
        lst90 = image.select('LST_Day_1km_90')

        tci30 = lst30.subtract(273.15).multiply(0.1).rename('TCI_30')
        tci60 = lst60.subtract(273.15).multiply(0.1).rename('TCI_60')
        tci90 = lst90.subtract(273.15).multiply(0.1).rename('TCI_90')

        return image.addBands([tci30, tci60, tci90])

    def extract_time_lagged_data(self, point: ee.Feature) -> ee.Image:
        """Extract time-lagged environmental variables for a point."""
        date = ee.Date(point.get("FINISHDATE"))
        point_geometry = point.geometry()

        # Define time lags (30, 60, 90 days before FINISHDATE)
        lags = {
            "30": date.advance(-30, "days"),
            "60": date.advance(-60, "days"),
            "90": date.advance(-90, "days")
        }

        # Function to compute mean/sum for a given lag
        def compute_variable(collection_id: str, bands: list, reducer: str, lag: str):
            reducer_fn = ee.Reducer.mean() if reducer == "mean" else ee.Reducer.sum()
            return (
                ee.ImageCollection(collection_id)
                .filterBounds(point_geometry)
                .filterDate(lags[lag], date)
                .select(bands)
                .reduce(reducer_fn)
                .rename(f"{bands[0]}_{lag}")
            )

        # Compute variables using the helper function
        modis_ndvi30 = compute_variable(
            "MODIS/061/MOD13A2", ["NDVI"], "mean", "30")
        modis_ndvi60 = compute_variable(
            "MODIS/061/MOD13A2", ["NDVI"], "mean", "60")
        modis_ndvi90 = compute_variable(
            "MODIS/061/MOD13A2", ["NDVI"], "mean", "90")
        modis_evi30 = compute_variable(
            "MODIS/061/MOD13A2", ["EVI"], "mean", "30")
        modis_evi60 = compute_variable(
            "MODIS/061/MOD13A2", ["EVI"], "mean", "60")
        modis_evi90 = compute_variable(
            "MODIS/061/MOD13A2", ["EVI"], "mean", "90")
        modis_lst30 = compute_variable(
            "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "30")
        modis_lst60 = compute_variable(
            "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "60")
        modis_lst90 = compute_variable(
            "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "90")
        chirps30 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "30")
        chirps60 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "60")
        chirps90 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "90")
        era5_u30 = compute_variable(
            "ECMWF/ERA5/DAILY", ["u_component_of_wind_10m"], "mean", "30")
        era5_u60 = compute_variable(
            "ECMWF/ERA5/DAILY", ["u_component_of_wind_10m"], "mean", "60")
        era5_u90 = compute_variable(
            "ECMWF/ERA5/DAILY", ["u_component_of_wind_10m"], "mean", "90")
        era5_v30 = compute_variable(
            "ECMWF/ERA5/DAILY", ["v_component_of_wind_10m"], "mean", "30")
        era5_v60 = compute_variable(
            "ECMWF/ERA5/DAILY", ["v_component_of_wind_10m"], "mean", "60")
        era5_v90 = compute_variable(
            "ECMWF/ERA5/DAILY", ["v_component_of_wind_10m"], "mean", "90")
        smap30 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "30")
        smap60 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "60")
        smap90 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "90")

        # ... (similar for other variables)

        # Combine all bands into a single image
        combined_image = ee.Image.cat([modis_ndvi30, modis_ndvi60, modis_ndvi90, modis_evi30, modis_evi60, modis_evi90, modis_lst30, modis_lst60,
                                      modis_lst90, chirps30, chirps60, chirps90, era5_u30, era5_u60, era5_u90, era5_v30, era5_v60, era5_v90, smap30, smap60, smap90])
        return self.calculate_vhi(self.calculate_tci(combined_image))

    def generate_pseudo_absence_points(self, locust_points: ee.FeatureCollection, aoi: ee.Geometry) -> ee.FeatureCollection:
        """Generate pseudo-absence points avoiding actual swarm locations"""
        # Ensure the AOI is a valid Earth Engine Geometry object
        if not isinstance(aoi, ee.Geometry):
            aoi = ee.Geometry(aoi)
        # Generate random points
        non_swarm_points = ee.FeatureCollection.randomPoints(
            region=aoi,
            points=3300,
            seed=42
        )

        num_absence = non_swarm_points.size().getInfo()
        logging.info(f"Generated {num_absence} pseudo-absence points")

        # Create 10km buffers around swarm points
        swarm_buffers = locust_points.map(lambda feature:
                                          feature.buffer(10000)  # 10 km buffer
                                          )

        buffer_union = swarm_buffers.union()

        # Filter out points that fall within swarm buffers and add presence property
        filtered_non_swarm = non_swarm_points \
            .filter(ee.Filter.bounds(buffer_union).Not()) \
            .map(lambda feature:
                 feature.set({
                     'LOCPRESENT': 0,  # Label as absence
                     # Current date as placeholder
                     'FINISHDATE': ee.Date(datetime.now().strftime('%Y-%m-%d'))
                 })
                 )

        return filtered_non_swarm

    def prepare_training_data(self, fao_report_asset_id: str, aoi: ee.Geometry) -> ee.FeatureCollection:
        """Prepare training data by combining presence and pseudo-absence points"""
        # Load FAO desert locust report
        locust_points = ee.FeatureCollection(fao_report_asset_id)

        # Ensure presence points have correct properties
        locust_points = locust_points.map(lambda feature:
                                          # Ensure presence is marked as 1
                                          feature.set('LOCPRESENT', 1)
                                          )

        presence_count = locust_points.size().getInfo()
        # Generate and get pseudo-absence points
        non_swarm_points = self.generate_pseudo_absence_points(
            locust_points, aoi)
        absence_count = non_swarm_points.size().getInfo()

        logging.info(
            f"Presence points: {presence_count}, Absence points: {absence_count}")

        # Merge presence and absence points
        return locust_points.merge(non_swarm_points)

    def create_export_task(self, feature_index: int, feature: ee.Feature, aoi: ee.Geometry) -> ee.batch.Task:
        """Create an export task for a single feature"""
        finish_date = ee.Date(feature.get('FINISHDATE')
                              ).format('YYYY-MM-dd').getInfo()
        presence = feature.get('LOCPRESENT').getInfo()
        # Extract time-lagged data
        time_lagged_data = self.extract_time_lagged_data(feature).toFloat()

        # Create multi-band image
        multi_band_image = ee.Image.cat([
            time_lagged_data,
            ee.Image.constant(feature.get('LOCPRESENT')
                              ).toFloat().rename('label')
        ]).clip(aoi)

        # Create export task using keyword arguments
        task = ee.batch.Export.image.toDrive(
            image=multi_band_image,
            description=f'et_locust_image_dataset_{finish_date}_presence_{presence}_{feature_index+1}',
            scale=self.common_scale,
            region=aoi,
            maxPixels=1e13,
            crs=self.common_projection,
            folder='Thesis'
        )

        return task

    def monitor_and_export(self, fao_report_asset_id: str, batch_size: int = 250):
        aoi = self.get_ethiopia_boundary()
        training_data = self.prepare_training_data(fao_report_asset_id, aoi)
        total_features = training_data.size().getInfo()
        num_batches = math.ceil(total_features / batch_size)

        active_tasks: Dict[str, ee.batch.Task] = {}
        completed_count = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            batch_features = training_data.toList(
                batch_size, start_idx).getInfo()

            for idx, feature in enumerate(batch_features):
                global_idx = start_idx + idx

                # Throttle task creation to avoid exceeding limits
                while len(active_tasks) >= 250:
                    self._cleanup_tasks(active_tasks)
                    time.sleep(60)

                # Create and start export task
                feature_ee = ee.Feature(feature)
                task = self.create_export_task(global_idx, feature_ee, aoi)
                task.start()
                active_tasks[f"task_{global_idx}"] = task
                logging.info(
                    f"Started export task for image {global_idx}/{total_features} of batch {batch_idx + 1}/{num_batches}")

            # Monitor tasks periodically
            while active_tasks:
                self._cleanup_tasks(
                    active_tasks, completed_count, total_features)
                time.sleep(60)  # Check every minute

        logging.info(f"All exports completed. Total: {completed_count}")

    def _cleanup_tasks(self, active_tasks: dict, completed_count: int = 0, total: int = 0):
        """Cleanup completed/failed tasks and update counters."""
        for task_id, task in list(active_tasks.items()):
            status = task.status()
            if status["state"] == "COMPLETED":
                del active_tasks[task_id]
                completed_count += 1
                if completed_count % 100 == 0:
                    logging.info(f"Progress: {completed_count}/{total}")
            elif status["state"] == "FAILED":
                logging.error(
                    f"Task {task_id} failed: {status.get('error_message')}")
                del active_tasks[task_id]

# Main function to initialize and run the exporter


def main(fao_report_asset_id: str):
    # Initialize GEEExporter instance
    exporter = GEEExporter()

    # Initialize Earth Engine
    exporter.initialize_ee()

    # Start the monitoring and exporting tasks
    exporter.monitor_and_export(fao_report_asset_id)


if __name__ == "__main__":
    # Example FAO report asset ID (replace with actual asset ID)
    fao_report_asset_id = 'projects/desert-locust-forcast/assets/FAO_Swarm_Report_RAW_2019_2021'
    main(fao_report_asset_id)
