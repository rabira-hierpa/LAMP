import streamlit as st
import ee
import time
import logging
from datetime import datetime
from google.oauth2 import service_account
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class GEEExporter:
    def __init__(self):
        self.common_scale = 1000  # ~1 km resolution
        self.common_projection = 'EPSG:4326'  # WGS84 projection

    def initialize_ee(self):
        """Initialize Earth Engine using credentials from Streamlit secrets."""
        try:
            service_account_keys = st.secrets["service_account"]
            credentials = service_account.Credentials.from_service_account_info(
                service_account_keys, scopes=ee.oauth.SCOPES)
            ee.Initialize(credentials)
            logging.info("Earth Engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Earth Engine: {str(e)}")
            raise

    def calculate_vhi(self, image: ee.Image) -> ee.Image:
        """
        Calculate Vegetation Health Index based on NDVI and TCI bands.
        VHI = 0.5*NDVI + 0.5*TCI.
        """
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

    def calculate_tci(self, image: ee.Image) -> ee.Image:
        """
        Calculate Temperature Condition Index from LST bands.
        Example: convert from Kelvin to Celsius, then scale by 0.1.
        TCI = (LST_K - 273.15) * 0.1
        """
        lst30 = image.select('LST_Day_1km_30')
        lst60 = image.select('LST_Day_1km_60')
        lst90 = image.select('LST_Day_1km_90')

        tci30 = lst30.subtract(273.15).multiply(0.1).rename('TCI_30')
        tci60 = lst60.subtract(273.15).multiply(0.1).rename('TCI_60')
        tci90 = lst90.subtract(273.15).multiply(0.1).rename('TCI_90')

        return image.addBands([tci30, tci60, tci90])

    def extract_time_lagged_data(self, point: ee.Feature) -> ee.Image:
        """
        For a given point (which has a FINISHDATE), calculate multiple 
        time-lagged environmental variables (30, 60, 90 days).
        Returns a multi-band image with NDVI, EVI, LST, precipitation,
        wind components, soil moisture, plus derived VHI and TCI.
        """
        date = ee.Date(point.get("FINISHDATE"))
        point_geometry = point.geometry()

        # Define time lags
        lags = {
            "30": date.advance(-30, "days"),
            "60": date.advance(-60, "days"),
            "90": date.advance(-90, "days")
        }

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

        # CHIRPS precipitation (daily sum)
        chirps30 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "30")
        chirps60 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "60")
        chirps90 = compute_variable(
            "UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "90")

        # ERA5 wind (daily means)
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

        # SMAP soil moisture (surface)
        smap30 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "30")
        smap60 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "60")
        smap90 = compute_variable(
            "NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "90")

        # Combine all bands
        combined_image = ee.Image.cat([
            modis_ndvi30, modis_ndvi60, modis_ndvi90,
            modis_evi30,  modis_evi60,  modis_evi90,
            modis_lst30,  modis_lst60,  modis_lst90,
            chirps30,     chirps60,     chirps90,
            era5_u30,     era5_u60,     era5_u90,
            era5_v30,     era5_v60,     era5_v90,
            smap30,       smap60,       smap90
        ])

        # Calculate TCI and VHI
        combined_image = self.calculate_tci(combined_image)
        combined_image = self.calculate_vhi(combined_image)

        return combined_image

    def generate_pseudo_absence_points(self, locust_points: ee.FeatureCollection) -> ee.FeatureCollection:
        """
        Generate pseudo-absence points within the bounding box of the presence points, 
        excluding a 10 km buffer around each presence point.
        """
        # Determine the bounding box of all locust_points
        presence_extent = locust_points.geometry().bounds()

        # Generate random points in that bounding box
        non_swarm_points = ee.FeatureCollection.randomPoints(
            region=presence_extent,
            points=3300,  # Adjust as needed
            seed=42
        )

        # Buffer each swarm point by 10 km, then union them
        swarm_buffers = locust_points.map(lambda f: f.buffer(10000))
        buffer_union = swarm_buffers.union()

        # Exclude random points that fall within 10 km of swarm points
        filtered_non_swarm = non_swarm_points \
            .filter(ee.Filter.bounds(buffer_union).Not()) \
            .map(
                lambda feature: feature.set({
                    'LOCPRESENT': 0,
                    # You can choose how FINISHDATE is set for pseudo-absence
                    'FINISHDATE': ee.Date(datetime.now().strftime('%Y-%m-%d'))
                })
            )

        return filtered_non_swarm

    def prepare_training_data(self, fao_report_asset_id: str) -> ee.FeatureCollection:
        """
        Load the FAO swarm report as presence points. Generate pseudo-absence
        in the bounding box of those presence points. Merge them into one collection.
        """
        # Mark FAO points as presence
        locust_points = ee.FeatureCollection(fao_report_asset_id).map(
            lambda f: f.set('LOCPRESENT', 1)
        )

        # Generate pseudo-absence points
        non_swarm_points = self.generate_pseudo_absence_points(locust_points)
        logging.info(f"Presence points: {locust_points.size().getInfo()}")
        logging.info(
            f"Pseudo-absence points: {non_swarm_points.size().getInfo()}")
        # Merge presence + pseudo-absence
        return locust_points.merge(non_swarm_points)

    def create_export_task(self, feature_index: int, feature: ee.Feature) -> ee.batch.Task:
        """
        Create an export task for a single feature. The region is a 10 km buffer
        around that feature's geometry.
        """
        finish_date = ee.Date(feature.get('FINISHDATE')
                              ).format('YYYY-MM-dd').getInfo()
        presence = feature.get('LOCPRESENT').getInfo()

        # Extract the multi-band image for 30/60/90-day lags
        time_lagged_data = self.extract_time_lagged_data(feature).toFloat()

        # Create a 10 km buffer geometry
        point_geometry = feature.geometry()
        patch_geometry = point_geometry.buffer(10000)

        # Add label band and clip to that 10 km patch
        multi_band_image = ee.Image.cat([
            time_lagged_data,
            ee.Image.constant(presence).toFloat().rename('label')
        ]).clip(patch_geometry)

        # Make a descriptive name for the export
        export_description = f"locust_{finish_date}_label_{presence}_{feature_index+1}"

        return ee.batch.Export.image.toDrive(
            image=multi_band_image,
            description=export_description,
            scale=self.common_scale,
            region=patch_geometry,
            maxPixels=1e13,
            crs=self.common_projection,
            folder='Thesis'
        )

    def monitor_and_export(self, fao_report_asset_id: str, max_concurrent_tasks: int = 250):
        """
        Convert training data to a list, create export tasks, and 
        manage them in a queue with concurrency control.
        """
        # 1) Prepare the presence & pseudo-absence dataset
        training_data = self.prepare_training_data(fao_report_asset_id)

        # 2) Convert FeatureCollection to a Python list
        all_features = training_data.toList(training_data.size()).getInfo()
        total_features = len(all_features)
        st.info(f"Found {total_features} features to export.")

        # 3) Prepare all export tasks
        task_queue = []
        for feature_index, feature_dict in enumerate(all_features):
            feature_ee = ee.Feature(feature_dict)
            task = self.create_export_task(feature_index, feature_ee)
            task_queue.append(task)

        active_tasks: List[ee.batch.Task] = []
        completed_count = 0

        # 4) Process tasks in a loop until queue is empty and all tasks finish
        while task_queue or active_tasks:
            # Start new tasks if we're under the concurrency limit
            while len(active_tasks) < max_concurrent_tasks and task_queue:
                task = task_queue.pop(0)
                task.start()
                status_info = task.status()
                task_desc = status_info.get(
                    'description', 'Task has no description')
                logging.info(f"Started export task: {task_desc}")
                st.write(f"🟡 Started task: {task_desc}")
                active_tasks.append(task)

            # Check status of all active tasks
            for task in list(active_tasks):
                status = task.status()
                state = status.get('state', 'UNKNOWN')
                task_id = status.get('id', 'Unknown ID')
                task_desc = status.get(
                    'description', 'Task has no description')

                # Log progress every time the number of active tasks is multiple of 250
                if len(active_tasks) % 250 == 0:
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

            # Sleep 30s to avoid hitting Earth Engine quotas with too-frequent checks
            time.sleep(30)

        # 5) Final success message
        logging.info(
            f"All exports completed. Total: {completed_count}/{total_features}")
        st.success(
            f"All exports completed! Total processed: {completed_count}")


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("Ethiopia Locust Data Export")

    # User-provided FAO report (swarm) asset ID
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
