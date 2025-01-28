// Load the FAO GAUL dataset (2015 version)
var gaulDataset = ee.FeatureCollection("FAO/GAUL/2015/level1");
var etBoundary = gaulDataset.filter(ee.Filter.eq('ADM0_NAME', 'Ethiopia'));

// Define area of interest and time range
var aoi = etBoundary.geometry();
var startDate = "2019-06-25";
var endDate = "2021-11-27";

var commonScale = 1000; // 1 km resolution
var commonProjection = 'EPSG:4326'; // WGS84 projection

// Function to reproject images to the common CRS and scale
function resampleAndReproject(image) {
  return image
    .resample('bilinear')
    .reproject({
      crs: commonProjection,
      scale: commonScale
    });
}

// Function to calculate VHI (Vegetation Health Index)
var calculateVHI = function(image) {
  var ndvi = image.select('NDVI');
  var tci = image.select('TCI'); // Temperature Condition Index
  var vhi = ndvi.multiply(0.5).add(tci.multiply(0.5)).rename('VHI');
  return image.addBands(vhi);
};

// Function to calculate TCI (Temperature Condition Index)
var calculateTCI = function(image) {
  var lst = image.select('LST_30'); // Land Surface Temperature
  var tci = lst.subtract(273.15).multiply(0.1).rename('TCI'); // Convert Kelvin to Celsius and scale
  return image.addBands(tci);
};

// Function to extract time-lagged variables
var extractTimeLaggedData = function(point) {
  var date = ee.Date(point.get('FINISHDATE')); // Ensure 'date' property exists
  var pointGeometry = point.geometry();

  // Define time lags (e.g., 30, 60, 90 days before the swarm date)
  var lag30 = date.advance(-30, 'days');
  var lag60 = date.advance(-60, 'days');
  var lag90 = date.advance(-90, 'days');

  // Extract MODIS NDVI with time lags
  var modisNDVI30 = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterBounds(pointGeometry)
    .filterDate(lag30, date)
    .select('NDVI')
    .mean()
    .rename('NDVI_30');

  var modisNDVI60 = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterBounds(pointGeometry)
    .filterDate(lag60, date)
    .select('NDVI')
    .mean()
    .rename('NDVI_60');

  var modisNDVI90 = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterBounds(pointGeometry)
    .filterDate(lag90, date)
    .select('NDVI')
    .mean()
    .rename('NDVI_90');

  // Extract MODIS EVI with time lags
  var modisEVI30 = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterBounds(pointGeometry)
    .filterDate(lag30, date)
    .select('EVI')
    .mean()
    .rename('EVI_30');

  var modisEVI60 = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterBounds(pointGeometry)
    .filterDate(lag60, date)
    .select('EVI')
    .mean()
    .rename('EVI_60');

  var modisEVI90 = ee.ImageCollection('MODIS/061/MOD13A2')
    .filterBounds(pointGeometry)
    .filterDate(lag90, date)
    .select('EVI')
    .mean()
    .rename('EVI_90');

  // Extract MODIS LST with time lags
  var modisLST30 = ee.ImageCollection('MODIS/061/MOD11A2')
    .filterBounds(pointGeometry)
    .filterDate(lag30, date)
    .select('LST_Day_1km')
    .mean()
    .rename('LST_30');

  var modisLST60 = ee.ImageCollection('MODIS/061/MOD11A2')
    .filterBounds(pointGeometry)
    .filterDate(lag60, date)
    .select('LST_Day_1km')
    .mean()
    .rename('LST_60');

  var modisLST90 = ee.ImageCollection('MODIS/061/MOD11A2')
    .filterBounds(pointGeometry)
    .filterDate(lag90, date)
    .select('LST_Day_1km')
    .mean()
    .rename('LST_90');

  // Extract CHIRPS precipitation with time lags
  var chirps30 = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterBounds(pointGeometry)
    .filterDate(lag30, date)
    .select('precipitation')
    .sum()
    .rename('precip_30');

  var chirps60 = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterBounds(pointGeometry)
    .filterDate(lag60, date)
    .select('precipitation')
    .sum()
    .rename('precip_60');

  var chirps90 = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterBounds(pointGeometry)
    .filterDate(lag90, date)
    .select('precipitation')
    .sum()
    .rename('precip_90');

  // Extract ERA5 u/v wind speed with time lags
  var era5u30 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filterBounds(pointGeometry)
    .filterDate(lag30, date)
    .select('u_component_of_wind_10m')
    .mean()
    .rename('u_wind_30');

  var era5u60 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filterBounds(pointGeometry)
    .filterDate(lag60, date)
    .select('u_component_of_wind_10m')
    .mean()
    .rename('u_wind_60');

  var era5u90 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filterBounds(pointGeometry)
    .filterDate(lag90, date)
    .select('u_component_of_wind_10m')
    .mean()
    .rename('u_wind_90');

  var era5v30 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filterBounds(pointGeometry)
    .filterDate(lag30, date)
    .select('v_component_of_wind_10m')
    .mean()
    .rename('v_wind_30');

  var era5v60 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filterBounds(pointGeometry)
    .filterDate(lag60, date)
    .select('v_component_of_wind_10m')
    .mean()
    .rename('v_wind_60');

  var era5v90 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    .filterBounds(pointGeometry)
    .filterDate(lag90, date)
    .select('v_component_of_wind_10m')
    .mean()
    .rename('v_wind_90');

  // Extract SMAP soil moisture with time lags
  var smap30 = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
    .filterBounds(pointGeometry)
    .filterDate(lag30, date)
    .select('sm_surface')
    .mean()
    .rename('soil_moisture_30');

  var smap60 = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
    .filterBounds(pointGeometry)
    .filterDate(lag60, date)
    .select('sm_surface')
    .mean()
    .rename('soil_moisture_60');

  var smap90 = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
    .filterBounds(pointGeometry)
    .filterDate(lag90, date)
    .select('sm_surface')
    .mean()
    .rename('soil_moisture_90');

  // Combine all time-lagged data into a single image
  var timeLaggedData = ee.Image.cat(
    modisNDVI30,
    modisNDVI60,
    modisNDVI90,
    modisEVI30,
    modisEVI60,
    modisEVI90,
    modisLST30,
    modisLST60,
    modisLST90,
    chirps30,
    chirps60,
    chirps90,
    era5u30,
    era5u60,
    era5u90,
    era5v30,
    era5v60,
    era5v90,
    smap30,
    smap60,
    smap90
  );

  // Calculate VHI and TCI
  var vhiTciImage = calculateVHI(calculateTCI(timeLaggedData));

  return vhiTciImage;
};

// Function to aggregate environmental data over a buffer zone
var aggregateOverBuffer = function(image, point, bufferRadius) {
  var pointGeometry = point.geometry().buffer(bufferRadius);
  var stats = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: pointGeometry,
    scale: commonScale,
    maxPixels: 1e13
  });
  return stats;
};

// Load FAO desert locust report (shapefile already uploaded)
var locustPoints = ee.FeatureCollection(faoReport);

// Ensure each feature has a valid 'date' property
var locustPointsWithDate = locustPoints.map(function(feature) {
  // If 'date' property is missing, add it (replace with your logic to extract the date)
  var date = feature.get('FINISHDATE'); // Replace 'date' with the actual property name
  if (!date) {
    date = '2019-06-25'; // Assign a default date if missing
  }
  return feature.set('FINISHDATE', ee.Date(date));
});

// Ensure each feature has a valid 'presence' property
var locustPointsWithPresence = locustPointsWithDate.map(function(feature) {
  // If 'presence' property is missing, add it (replace with your logic to extract the presence)
  var presence = feature.get('LOCPRESENCE'); // Replace 'presence' with the actual property name
  if (!presence) {
    presence = 1; // Assign a default value (1 for presence, 0 for absence)
  }
  return feature.set('LOCPRESENCE', presence);
});

// Generate pseudo-absence points (0 for absence)
var nonSwarmPoints = ee.FeatureCollection.randomPoints({
  region: aoi,
  points: 1000, // Adjust number of points
  seed: 42 // Random seed for reproducibility
});

// Exclude pseudo-absence points within 10 km of swarm points
var swarmBuffers = locustPointsWithPresence.map(function(feature) {
  return feature.buffer(10000); // 10 km buffer
});
var nonSwarmPoints = nonSwarmPoints.filterBounds(swarmBuffers.union()).map(function(feature) {
  return feature.set('presence', 0); // Label as absence
});

// Combine presence and absence points
var trainingData = locustPointsWithPresence.merge(nonSwarmPoints);

// Extract the first feature from the FeatureCollection
var firstFeature = ee.Feature(trainingData.first());

// Print the first feature to inspect its properties
print('First Feature:', firstFeature);

// Extract time-lagged data for the first feature
var timeLaggedData = extractTimeLaggedData(firstFeature);

// Aggregate environmental data over a 10 km buffer for the first feature
var aggregatedData = aggregateOverBuffer(timeLaggedData, firstFeature, 10000); // 10 km buffer

// Print the aggregated data to inspect the results
print('Aggregated Data for First Feature:', aggregatedData);

// Create a multi-band image from the aggregated data
var multiBandImage = ee.Image.cat([
  timeLaggedData, // Time-lagged variables
  ee.Image.constant(firstFeature.get('LOCPRESENT')).rename('label') // Add label band
]);

// Add the multi-band image to the map for visualization
Map.addLayer(multiBandImage, {bands: ['NDVI_30'], min: 0, max: 1, palette: ['red', 'yellow', 'green']}, 'NDVI 30 Days');
Map.addLayer(multiBandImage, {bands: ['precip_30'], min: 0, max: 100, palette: ['blue', 'cyan', 'green']}, 'Precip 30 Days');

// Export the multi-band image as a GeoTIFF
Export.image.toDrive({
  image: multiBandImage,
  description: 'single_feature_multi_band_image_test',
  scale: commonScale,
  region: firstFeature.geometry().buffer(10000).bounds(), // Export the buffer region
  maxPixels: 1e13,
  crs: commonProjection,
  fileFormat: 'GeoTIFF',
  folder: 'Thesis'
});
