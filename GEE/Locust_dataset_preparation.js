// Load the FAO GAUL dataset (2015 version)
var gaulDataset = ee.FeatureCollection("FAO/GAUL/2015/level1");
var etBoundary = gaulDataset.filter(ee.Filter.eq("ADM0_NAME", "Ethiopia"));

// Define area of interest and time range
var aoi = etBoundary.geometry();
var startDate = "2019-06-25";
var endDate = "2021-11-27";

var commonScale = 30;
var commonProjection = ee
  .ImageCollection("NASA/HLS/HLSL30/v002")
  .first()
  .projection();

// Function to reproject images to the common CRS and scale
function resampleAndReproject(image, refBand) {
  return image.reproject({
    crs: refBand.projection(),
    scale: commonScale,
  });
}

// Function to print projection and scale information
function printProjectionInfo(image, name) {
  print(name + " Projection: ", image.projection());
  print(name + " CRS: ", image.projection().crs());
  print(name + " Scale: ", image.projection().nominalScale());
}

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
  .median()
  .clip(aoi);

// Use NDVI as the reference band for reprojecting other datasets
var refBand = hlsWithIndices.select("NDVI");

// 2. Soil Moisture (SMAP)
var soilMoisture = ee
  .ImageCollection("NASA/SMAP/SPL4SMGP/007")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("sm_surface")
  .median()
  .clip(aoi)
  .resample("bilinear");

printProjectionInfo(soilMoisture, "Soil Moisture");
// 3. Land Surface Temperature (MODIS)
var lst = ee
  .ImageCollection("MODIS/061/MOD11A2")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("LST_Day_1km")
  .map(function (image) {
    return image.multiply(0.02).subtract(273.15).rename("LST_Day_1km");
  })
  .median()
  .clip(aoi)
  .resample("bilinear");

printProjectionInfo(lst, "Land Surface Temperatrue");
// 4. Relative Humidity (ERA5-Land)
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
  .median()
  .clip(aoi)
  .resample("bilinear");

printProjectionInfo(relativeHumidity, "Relative humidity");
// 5. Elevation (SRTM)
var elevation = ee
  .Image("USGS/SRTMGL1_003")
  .select("elevation")
  .clip(aoi)
  .resample("bilinear");

printProjectionInfo(elevation, "Elevation");
// 6. Windspeed at 10m and 50m (ERA5-Land)
var vWindSpeed10m = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("v_component_of_wind_10m")
  .median()
  .clip(aoi)
  .resample("bilinear");

printProjectionInfo(vWindSpeed10m, "Windspeed 10");
var uWindSpeed = ee
  .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("u_component_of_wind_10m")
  .median()
  .clip(aoi)
  .resample("bilinear");

printProjectionInfo(uWindSpeed, "Windspeed u10");
// 7. Precipitation (CHIRPS)
var chirps = ee
  .ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .select("precipitation")
  .reduce(ee.Reducer.sum()) // Monthly sum
  .clip(aoi)
  .resample("bilinear");

printProjectionInfo(chirps, "Precipitation (CHIRPS)");
// Combine all datasets into a single image (only selected bands)
var combinedImage = hlsWithIndices
  .addBands(soilMoisture.rename("sm_surface"))
  .addBands(lst.rename("LST_Day_1km"))
  .addBands(relativeHumidity.rename("Relative_Humidity"))
  .addBands(uWindSpeed.rename("uWindSpeed_10m"))
  .addBands(vWindSpeed10m.rename("vWindSpeed_10m"))
  .addBands(chirps.rename("precipitation"))
  .addBands(elevation.rename("elevation"));

// Load FAO desert locust report (shapefile already uploaded)
var locustPoints = ee.FeatureCollection(faoReport);

// Create a mask for locust presence
var locustPresence = locustPoints
  .map(function (feature) {
    return feature.set("presence", 1);
  })
  .reduceToImage({
    properties: ["presence"],
    reducer: ee.Reducer.first(),
  })
  .unmask(0)
  .rename("label");

// Add locust presence to the combined environmental data
var labeledDataWithLocation = combinedImage.addBands(locustPresence);

// Add latitude and longitude bands to labeledData
var latLon = ee.Image.pixelLonLat().clip(aoi);
labeledDataWithLocation = labeledDataWithLocation
  .addBands(latLon.select("longitude"))
  .addBands(latLon.select("latitude"));

// Split into 4 batches (3303 points total, 825 points per batch)
var totalPoints = locustPoints.size(); // Total number of points
var batchSize = ee.Number(825); // Approximate number of points per batch

// Function to get a batch of points
function getBatch(startIndex) {
  return locustPoints.toList(batchSize, startIndex); // Get 'batchSize' points starting from 'startIndex'
}

// Get each batch of points
var batch1 = ee.FeatureCollection(getBatch(0));
var batch2 = ee.FeatureCollection(getBatch(825));
var batch3 = ee.FeatureCollection(getBatch(1650));
var batch4 = ee.FeatureCollection(getBatch(2475)); // Remaining points

function sampleAndExportBatch(batch, batchNumber) {
  var trainingSamplesBatch = labeledDataWithLocation.sampleRegions({
    collection: batch,
    scale: 30, // Adjust scale as needed
    geometries: true, // Keep geometries for analysis
  });

  // Export the data to Google Drive for external model training (in batches)
  Export.table.toDrive({
    collection: trainingSamplesBatch,
    description: "locust_training_data_batch" + batchNumber,
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
      "precipitation",
      "elevation",
      "longitude", // Include longitude
      "latitude", // Include latitude
      "label", // Locust presence/absence
    ],
  });
}

// Export each batch individually
sampleAndExportBatch(batch1, 1);
sampleAndExportBatch(batch2, 2);
sampleAndExportBatch(batch3, 3);
sampleAndExportBatch(batch4, 4);
