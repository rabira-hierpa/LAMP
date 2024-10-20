// Define area of interest and time range
var gaulDataset = ee.FeatureCollection("FAO/GAUL/2015/level1");
var etBoundary = gaulDataset.filter(ee.Filter.eq("ADM0_NAME", "Ethiopia"));
var aoi = etBoundary.geometry(); // area of interest
var startDate = "2019-06-25";
var endDate = "2021-11-27";

// 1. Load HLS dataset for NDVI and EVI (Sentinel-2 and Landsat)
var hlsL30Collection = ee
  .ImageCollection("NASA/HLS/HLSL30/v002")
  .filterBounds(aoi)
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.lt("CLOUD_COVERAGE", 30));

// Function to calculate NDVI and EVI
var addNDVI = function (image) {
  var ndvi = image.normalizedDifference(["B5", "B4"]).rename("NDVI"); // Red, NIR
  return image.addBands(ndvi);
};
var addEVI = function (image) {
  var evi = image
    .expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
      NIR: image.select("B5"),
      RED: image.select("B4"),
      BLUE: image.select("B2"),
    })
    .rename("EVI");
  return image.addBands(evi);
};
var hlsWithIndices = hlsL30Collection.map(addNDVI).map(addEVI);

// 2. Soil Moisture (SMAP)
var soilMoisture = ee
  .ImageCollection("NASA/SMAP/SPL4SMGP/007")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("sm_surface");

// 3. Land Surface Temperature (MODIS)
var lst = ee
  .ImageCollection("MODIS/061/MOD11A2")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("LST_Day_1km")
  .map(function (image) {
    return image.multiply(0.02).subtract(273.15).rename("LST_Day_1km"); // Convert to Celsius
  });

// 4. Relative Humidity (ERA5-Land)
var relativeHumidity = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select(["dewpoint_temperature_2m", "temperature_2m"])
  .map(function (image) {
    var temp = image.select("temperature_2m").subtract(273.15); // to Celsius
    var dewpoint = image.select("dewpoint_temperature_2m").subtract(273.15); // to Celsius
    var rh = ee
      .Image(100)
      .multiply(ee.Image(112).subtract(temp.subtract(dewpoint).multiply(5)))
      .exp()
      .divide(100)
      .rename("Relative_Humidity");
    return rh;
  });

// 5. Elevation (SRTM)
var elevation = ee.Image("USGS/SRTMGL1_003").clip(aoi);

// 6. Windspeed at 10m and 50m, and uWindSpeed (ERA5-Land)
var vWindSpeed10m = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("v_component_of_wind_10m");

var uWindSpeed = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("u_component_of_wind_10m");

// 7. Precipitation (CHIRPS)
var chirps = ee
  .ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("precipitation");

// Monthly precipitation sum from CHIRPS
var chirpsMonthly = ee
  .ImageCollection(
    ee.List.sequence(
      0,
      ee.Date(endDate).difference(ee.Date(startDate), "month").subtract(1)
    ).map(function (monthOffset) {
      var start = ee.Date(startDate).advance(monthOffset, "month");
      var end = start.advance(1, "month");

      var monthlySum = chirps
        .filterDate(start, end)
        .reduce(ee.Reducer.sum())
        .set("system:time_start", start.millis()); // Set time property to start of the month

      return monthlySum;
    })
  )
  .median()
  .rename("precipitation_monthly");

// Combine all datasets into a single image (median composite)
var combinedImage = hlsWithIndices
  .median()
  .addBands(soilMoisture.median())
  .addBands(lst.median())
  .addBands(relativeHumidity.median())
  .addBands(uWindSpeed.median().rename("uWindSpeed_10m"))
  .addBands(vWindSpeed10m.median().rename("vWindSpeed_10m"))
  .addBands(chirpsMonthly)
  .addBands(elevation);

// Load FAO desert locust report (shapefile already uploaded)
var locustPresence = ee.FeatureCollection(faoReport);

// Create a binary mask: 1 for locust presence, 0 for absence
var locustPresenceImage = locustPresence
  .map(function (feature) {
    return feature.set("label", 1);
  })
  .reduceToImage({
    properties: ["label"],
    reducer: ee.Reducer.first(),
  })
  .unmask(0); // Unmask with 0 for locust absence

// Add the locust label to the combined dataset
var labeledData = combinedImage.addBands(locustPresenceImage.rename("label"));

// Sample data with labels for training
var trainingSamples = labeledData.sample({
  region: faoReport,
  scale: 3000, // Match Sentinel-2 resolution
  // numPixels: 1000, // Adjust this based on your region size
  //seed: 42,
  geometries: true, // Keep geometries for analysis if needed
});

// Clip datasets to the AOI
var ndviClipped = hlsWithIndices.select("NDVI").median().clip(aoi);
var eviClipped = hlsWithIndices.select("EVI").median().clip(aoi);
var lstClipped = lst.median().clip(aoi);
var windspeed10mClipped = uWindSpeed.median().clip(aoi);
var locustPresenceClipped = locustPresence.filterBounds(aoi); // Clip the locust presence shapefile

// // Add layers to the map within the AOI
Map.centerObject(aoi, 6); // Center the map on the AOI
// Visualize NDVI median
//Map.addLayer(ndviClipped, {min: -1, max: 1, palette: ['blue', 'white', 'green']}, 'NDVI (AOI)');
// Visualize EVI median
//Map.addLayer(eviClipped, {min: -1, max: 1, palette: ['blue', 'white', 'green']}, 'EVI (AOI)');
// Visualize Land Surface Temperature (LST)
//Map.addLayer(lstClipped, {min: 20, max: 40, palette: ['blue', 'green', 'red']}, 'LST (AOI)');
// Visualize Windspeed at 10m
//Map.addLayer(windspeed10mClipped, {min: 0, max: 15, palette: ['blue', 'yellow', 'red']}, 'Windspeed at 10m (AOI)');
// Visualize Locust Presence areas
Map.addLayer(locustPresenceClipped, { color: "red" }, "Locust Presence (AOI)");

// Export the data to Google Drive for external model training
Export.table.toDrive({
  collection: trainingSamples,
  description: "locust_training_data",
  fileFormat: "CSV",
  scale: 3000,
  folder: "Thesis/Data/GEE", // Optional: Specify a folder in Drive
  selectors: [
    "NDVI",
    "EVI",
    "SSM",
    "LST_Day_1km",
    "Relative_Humidity",
    "Windspeed_10m",
    "vWindSpeed_10m",
    "uWindSpeed_10m",
    "precipitation_monthly",
    "elevation",
    "label",
  ], //Specify the attributes to export
});
// Export.table.toDrive(trainingSamples, "Locust_training");
