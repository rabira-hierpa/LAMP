import streamlit as st
import ee
import time
import logging
from datetime import datetime
from google.oauth2 import service_account
import math
from typing import Dict

# -----------------------------
# 1) Set up logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# -----------------------------
# 2) Define the GEEExporter class
# -----------------------------
class GEEExporter:
    def __init__(self):
        self.common_scale = 1000  # 1 km resolution
        self.common_projection = 'EPSG:4326'  # WGS84 projection

    def initialize_ee(self):
        """Initialize Earth Engine"""
        try:
            service_account_keys = st.secrets["service_account"]
            credentials = service_account.Credentials.from_service_account_info(
                service_account_keys, scopes=ee.oauth.SCOPES)
            ee.Initialize(credentials)
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

        # Helper function to compute mean/sum for a given lag
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

        # MODIS NDVI
        modis_ndvi30 = compute_variable(
            "MODIS/061/MOD13A2", ["NDVI"], "mean", "30")
        modis_ndvi60 = compute_variable(
            "MODIS/061/MOD13A2", ["NDVI"], "mean", "60")
        modis_ndvi90 = compute_variable(
            "MODIS/061/MOD13A2", ["NDVI"], "mean", "90")

        # MODIS EVI
        modis_evi30 = compute_variable(
            "MODIS/061/MOD13A2", ["EVI"], "mean", "30")
        modis_evi60 = compute_variable(
            "MODIS/061/MOD13A2", ["EVI"], "mean", "60")
        modis_evi90 = compute_variable(
            "MODIS/061/MOD13A2", ["EVI"], "mean", "90")

        # MODIS LST
        modis_lst30 = compute_variable(
            "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "30")
        modis_lst60 = compute_variable(
            "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "60")
        modis_lst90 = compute_variable(
            "MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "90")

        # CHIRPS precipitation
        chirps30 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "30")
        chirps60 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "60")
        chirps90 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "90")

        # ERA5 wind components
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

        # SMAP soil moisture
        smap30 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "30")
        smap60 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "60")
        smap90 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "90")

        # Combine all bands
        combined_image = ee.Image.cat([
            modis_ndvi30, modis_ndvi60, modis_ndvi90,
            modis_evi30, modis_evi60, modis_evi90,
            modis_lst30, modis_lst60, modis_lst90,
            chirps30, chirps60, chirps90,
            era5_u30, era5_u60, era5_u90,
            era5_v30, era5_v60, era5_v90,
            smap30, smap60, smap90
        ])

        # Calculate indices
        return self.calculate_vhi(self.calculate_tci(combined_image))

    def generate_pseudo_absence_points(self, locust_points: ee.FeatureCollection, aoi: ee.Geometry) -> ee.FeatureCollection:
        """Generate pseudo-absence points avoiding actual swarm locations."""
        # Generate random points
        non_swarm_points = ee.FeatureCollection.randomPoints(
            region=aoi,
            points=3300,  # Adjust number as needed
            seed=42
        )

        # Create 10km buffers around swarm points
        swarm_buffers = locust_points.map(
            lambda feature: feature.buffer(10000))
        buffer_union = swarm_buffers.union()

        # Filter out points that fall within swarm buffers and add presence property
        filtered_non_swarm = non_swarm_points \
            .filter(ee.Filter.bounds(buffer_union).Not()) \
            .map(lambda feature:
                 feature.set({
                     'LOCPRESENT': 0,
                     'FINISHDATE': ee.Date(datetime.now().strftime('%Y-%m-%d'))
                 })
                 )

        return filtered_non_swarm

    def prepare_training_data(self, fao_report_asset_id: str, aoi: ee.Geometry) -> ee.FeatureCollection:
        """Prepare training data by combining presence and pseudo-absence points."""
        locust_points = ee.FeatureCollection(fao_report_asset_id).map(
            lambda f: f.set('LOCPRESENT', 1)
        )

        # Generate pseudo-absence
        non_swarm_points = self.generate_pseudo_absence_points(
            locust_points, aoi)

        return locust_points.merge(non_swarm_points)

    def create_export_task(self, feature_index: int, feature: ee.Feature, aoi: ee.Geometry) -> ee.batch.Task:
        """Create an export task for a single feature."""
        finish_date = ee.Date(feature.get('FINISHDATE')
                              ).format('YYYY-MM-dd').getInfo()
        presence = feature.get('LOCPRESENT').getInfo()

        # Create the multi-band image
        time_lagged_data = self.extract_time_lagged_data(feature).toFloat()
        multi_band_image = ee.Image.cat([
            time_lagged_data,
            ee.Image.constant(feature.get('LOCPRESENT')
                              ).toFloat().rename('label')
        ]).clip(aoi)

        # Export
        return ee.batch.Export.image.toDrive(
            image=multi_band_image,
            description=f'et_locust_image_dataset_{finish_date}_presence_{presence}_{feature_index+1}',
            scale=self.common_scale,
            region=aoi,
            maxPixels=1e13,
            crs=self.common_projection,
            folder='Thesis'
        )

    def monitor_and_export(self, fao_report_asset_id: str, max_concurrent_tasks: int = 250):
        """Run the export process with queue management and concurrency control."""
        aoi = self.get_ethiopia_boundary()
        training_data = self.prepare_training_data(fao_report_asset_id, aoi)

        # Convert FeatureCollection to list of features
        all_features = training_data.toList(training_data.size()).getInfo()
        total_features = len(all_features)
        st.info(f"Found {total_features} features to export.")

        # Prepare all export tasks
        task_queue = []
        for feature_index, feature in enumerate(all_features):
            feature_ee = ee.Feature(feature)
            task = self.create_export_task(feature_index, feature_ee, aoi)
            task_queue.append(task)

        active_tasks: List[ee.batch.Task] = []
        completed_count = 0

        # Process tasks with queue system
        while task_queue or active_tasks:
            # Start tasks up to max concurrency
            while len(active_tasks) < max_concurrent_tasks and task_queue:
                task = task_queue.pop(0)
                task.start()
                active_tasks.append(task)
                task_desc = task.status().get('description', 'Unknown task')
                logging.info(f"Started export task: {task_desc}")
                st.write(f"🟡 Started task: {task_desc}")

            # Check task statuses
            for task in list(active_tasks):
                status = task.status()
                state = status.get('state', 'UNKNOWN')
                task_id = status.get('id', 'Unknown ID')
                task_desc = status.get('description', 'Unknown task')
                if active_tasks % 250 == 0:
                    logging.info(
                        f"🚧 Export progress: {completed_count}/{total_features}")
                    st.write(
                        f"🚧 Export progress: {completed_count}/{total_features}")
                if state == 'COMPLETED':
                    active_tasks.remove(task)
                    completed_count += 1
                    logging.info(
                        f"✅ Export completed: {task_desc} (ID: {task_id})")
                    st.success(
                        f"✅ Completed {completed_count}/{total_features}: {task_desc}")
                elif state in ['FAILED', 'CANCELED']:
                    active_tasks.remove(task)
                    error_msg = status.get('error_message', 'No error message')
                    logging.error(
                        f"❌ Export failed: {task_desc} - {error_msg}")
                    st.error(f"❌ Failed: {task_desc} - {error_msg}")

            # Throttle API calls
            time.sleep(30)

        logging.info(
            f"All exports completed. Total: {completed_count}/{total_features}")
        st.success(
            f"All exports completed! Total processed: {completed_count}")

# -----------------------------
# 3) Streamlit UI
# -----------------------------


def main():
    st.title("Ethiopia Locust Data Export")

    # User inputs
    fao_report_asset_id = st.text_input(
        "FAO Report Asset ID",
        "projects/desert-locust-forcast/assets/FAO_Swarm_Report_RAW_2019_2021"
    )
    max_concurrent = st.number_input("Max Concurrent Tasks", 1, 250, 250)

    if st.button("Run Export"):
        exporter = GEEExporter()
        with st.spinner("Initializing Earth Engine..."):
            try:
                exporter.initialize_ee()
            except Exception as e:
                st.error(f"Failed to initialize EE: {e}")
                return

        with st.spinner("Exporting..."):
            try:
                exporter.monitor_and_export(
                    fao_report_asset_id,
                    max_concurrent_tasks=max_concurrent
                )
            except Exception as e:
                st.error(f"Export error: {str(e)}")


if __name__ == "__main__":
    main()
