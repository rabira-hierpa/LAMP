// Define area of interest and time range
var aoi = oromiaBoundary.geometry();
var startDate = "2021-01-01";
var endDate = "2021-11-27";

// 1. Load HLS dataset for NDVI and EVI (Sentinel-2 and Landsat)
var hlsL30Collection = ee
  .ImageCollection("NASA/HLS/HLSL30/v002")
  .filterBounds(aoi)
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.lt("CLOUD_COVERAGE", 30));

// Function to calculate NDVI and EVI
var addNDVI = function (image) {
  var ndvi = image.normalizedDifference(["B5", "B4"]).rename("NDVI");
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

// Compute NDVI and EVI, then select only the NDVI and EVI bands
var hlsWithIndices = hlsL30Collection
  .map(addNDVI)
  .map(addEVI)
  .select(["NDVI", "EVI"])
  .median(); // Use median for composite

// 2. Soil Moisture (SMAP) - Select only the needed band
var soilMoisture = ee
  .ImageCollection("NASA/SMAP/SPL4SMGP/007")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("sm_surface") // Select the soil moisture band
  .median() // Use median composite
  .clip(aoi);

// 3. Land Surface Temperature (MODIS) - Select only LST band
var lst = ee
  .ImageCollection("MODIS/061/MOD11A2")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("LST_Day_1km") // Select only the LST Day band
  .map(function (image) {
    return image.multiply(0.02).subtract(273.15).rename("LST_Day_1km");
  })
  .median() // Use median composite
  .clip(aoi);

// 4. Relative Humidity (ERA5-Land) - Select only necessary bands
var relativeHumidity = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select(["dewpoint_temperature_2m", "temperature_2m"])
  .map(function (image) {
    var temp = image.select("temperature_2m").subtract(273.15); // Convert to Celsius
    var dewpoint = image.select("dewpoint_temperature_2m").subtract(273.15); // Convert to Celsius
    var rh = ee
      .Image(100)
      .multiply(ee.Image(112).subtract(temp.subtract(dewpoint).multiply(5)))
      .exp()
      .divide(100)
      .rename("Relative_Humidity");
    return rh;
  })
  .median() // Use median composite
  .clip(aoi);

// 5. Elevation (SRTM) - Select only the elevation band
var elevation = ee
  .Image("USGS/SRTMGL1_003")
  .select("elevation") // Select only the elevation band
  .reduceResolution({
    reducer: ee.Reducer.mean(),
    bestEffort: true,
  })
  .clip(aoi); // Clip to AOI

// 6. Windspeed at 10m and 50m (ERA5-Land) - Select only the needed bands
var vWindSpeed10m = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("v_component_of_wind_10m")
  .median() // Use median composite
  .clip(aoi);

var uWindSpeed = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("u_component_of_wind_10m")
  .median() // Use median composite
  .clip(aoi);

// 7. Precipitation (CHIRPS) - Select only the precipitation band
var chirps = ee
  .ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("precipitation") // Select only the precipitation band
  .reduce(ee.Reducer.sum()) // Monthly sum
  .clip(aoi);

// Combine all datasets into a single image (only selected bands)
var combinedImage = hlsWithIndices
  .addBands(soilMoisture)
  .addBands(lst)
  .addBands(relativeHumidity)
  .addBands(uWindSpeed.rename("uWindSpeed_10m"))
  .addBands(vWindSpeed10m.rename("vWindSpeed_10m"))
  .addBands(chirps.rename("precipitation_monthly"))
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

// Add Latitude and Longitude bands to labeledData
var latLon = ee.Image.pixelLonLat().clip(aoi);
var labeledDataWithLocation = labeledData
  .addBands(latLon.select("longitude"))
  .addBands(latLon.select("latitude"));

// Divide the area into tiles (batch sampling)
// Define a 30km grid size
var gridSize = 30000; // Grid size in meters (30 km)
var grid = ee
  .FeatureCollection(aoi)
  .geometry()
  .buffer(gridSize / 2)
  .bounds()
  .coveringGrid("EPSG:4326", gridSize); // Generate 30km grid cells

// Sample data in each grid cell
var samples = grid.map(function (tile) {
  return labeledDataWithLocation.sample({
    region: tile.geometry(),
    scale: 300, // Match Sentinel-2 resolution
    numPixels: 500, // Fewer pixels per tile to reduce memory
    seed: 42,
    geometries: true,
  });
});

// Flatten the sample collection from all tiles
var trainingSamples = samples.flatten();

// Export the data to Google Drive for external model training (in batches)
Export.table.toDrive({
  collection: trainingSamples,
  description: "locust_training_data_300m_oromia_chunk",
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
    "longitude", // Include longitude
    "latitude", // Include latitude
    "label",
  ],
});
