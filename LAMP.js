// Define area of interest and time range
var aoi = oromiaBoundary.geometry();
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

// Reduced temporal resolution by averaging the collection
var hlsWithIndices = hlsL30Collection.map(addNDVI).map(addEVI).mean(); // Use mean to reduce memory

// 2. Soil Moisture (SMAP) - Clipped early to reduce memory
var soilMoisture = ee
  .ImageCollection("NASA/SMAP/SPL4SMGP/007")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("sm_surface")
  .map(function (image) {
    return image.clip(aoi);
  }); // Clip to AOI early

// 3. Land Surface Temperature (MODIS) - Clipped early to reduce memory
var lst = ee
  .ImageCollection("MODIS/061/MOD11A2")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("LST_Day_1km")
  .map(function (image) {
    return image
      .multiply(0.02)
      .subtract(273.15)
      .rename("LST_Day_1km")
      .clip(aoi);
  });

// 4. Relative Humidity (ERA5-Land) - No changes
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

// 5. Elevation (SRTM) - No changes
var elevation = ee.Image("USGS/SRTMGL1_003").clip(aoi);

// 6. Windspeed at 10m and 50m (ERA5-Land) - Clipped early to reduce memory
var vWindSpeed10m = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("v_component_of_wind_10m")
  .map(function (image) {
    return image.clip(aoi);
  }); // Clip to AOI early

var uWindSpeed = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("u_component_of_wind_10m")
  .map(function (image) {
    return image.clip(aoi);
  }); // Clip to AOI early

// 7. Precipitation (CHIRPS) - Simplify the temporal composite and clip
var chirps = ee
  .ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("precipitation")
  .map(function (image) {
    return image.clip(aoi);
  }); // Clip to AOI early

// Monthly precipitation sum from CHIRPS - Adjusted for memory
var chirpsMonthly = chirps
  .reduce(ee.Reducer.sum()) // Simplified precipitation sum
  .rename("precipitation_monthly");

// Combine all datasets into a single image
var combinedImage = hlsWithIndices
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

// Reduce the number of samples to avoid memory issues
var trainingSamples = labeledData.sample({
  region: aoi,
  scale: 300,
  numPixels: 1000, // Reduced to avoid memory issues
  seed: 42,
  geometries: true,
});

// Clip datasets to the AOI
var ndviClipped = hlsWithIndices.select("NDVI").clip(aoi);
var eviClipped = hlsWithIndices.select("EVI").clip(aoi);
var lstClipped = lst.median().clip(aoi);
var windspeed10mClipped = uWindSpeed.median().clip(aoi);
var locustPresenceClipped = locustPresence.filterBounds(aoi);

// Add layers to the map within the AOI
Map.centerObject(aoi, 6);
Map.addLayer(
  ndviClipped,
  { min: -1, max: 1, palette: ["blue", "white", "green"] },
  "NDVI (AOI)",
);
Map.addLayer(
  eviClipped,
  { min: -1, max: 1, palette: ["blue", "white", "green"] },
  "EVI (AOI)",
);
Map.addLayer(
  lstClipped,
  { min: 20, max: 40, palette: ["blue", "green", "red"] },
  "LST (AOI)",
);
Map.addLayer(
  windspeed10mClipped,
  { min: 0, max: 15, palette: ["blue", "yellow", "red"] },
  "Windspeed at 10m (AOI)",
);
Map.addLayer(
  locustPresenceClipped,
  { color: "white" },
  "Locust Presence (AOI)",
);

// Export the data to Google Drive for external model training
Export.table.toDrive({
  collection: trainingSamples,
  description: "locust_training_data_300m_oromia",
  fileFormat: "CSV",
  folder: "Thesis",
  selectors: [
    "NDVI",
    "EVI",
    "sm_surface",
    "LST_Day_1km",
    "Relative_Humidity",
    "uWindSpeed_10m",
    "vWindSpeed_10m",
    "precipitation_monthly",
    "elevation",
    "label",
  ],
});
