// Load the FAO GAUL dataset (2015 version)
var gaulDataset = ee.FeatureCollection("FAO/GAUL/2015/level1");
var etBoundary = gaulDataset.filter(ee.Filter.eq("ADM0_NAME", "Ethiopia"));

// Define area of interest
var aoi = etBoundary.geometry();

// Load FAO desert locust report and get the first feature (assuming 'timestamp' is a property)
var locustPoints = ee.FeatureCollection(faoReport);
var firstReport = ee.Feature(locustPoints.first()); // Retrieve the first locust report

// Function to get environmental data for a locust report's timestamp
function getEnvironmentDataImageForTimestamp(report) {
  var reportDate = ee.Date(report.get("FINISHDATE"));

  // Define a ±7-day time window around the report date
  var start = reportDate.advance(-7, "day");
  var end = reportDate.advance(7, "day");

  // 1. NDVI and EVI (HLS)
  var hlsImage = ee
    .ImageCollection("NASA/HLS/HLSL30/v002")
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.lt("CLOUD_COVERAGE", 30))
    .map(function (image) {
      var ndvi = image.normalizedDifference(["B5", "B4"]).rename("NDVI");
      var evi = image
        .expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
          NIR: image.select("B5"),
          RED: image.select("B4"),
          BLUE: image.select("B2"),
        })
        .rename("EVI");
      return image.addBands(ndvi).addBands(evi);
    })
    .select(["NDVI", "EVI"])
    .median()
    .clip(aoi);

  // 2. Soil Moisture (SMAP)
  var soilMoistureImage = ee
    .ImageCollection("NASA/SMAP/SPL4SMGP/007")
    .filterBounds(aoi)
    .filterDate(start, end)
    .select("sm_surface")
    .median()
    .clip(aoi);

  // 3. Land Surface Temperature (MODIS)
  var lstImage = ee
    .ImageCollection("MODIS/061/MOD11A2")
    .filterBounds(aoi)
    .filterDate(start, end)
    .select("LST_Day_1km")
    .map(function (image) {
      return image.multiply(0.02).subtract(273.15).rename("LST_Day_1km");
    })
    .median()
    .clip(aoi);

  // 4. Relative Humidity (ERA5-Land)
  var humidityImage = ee
    .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(aoi)
    .filterDate(start, end)
    .select(["dewpoint_temperature_2m", "temperature_2m"])
    .map(function (image) {
      var temp = image.select("temperature_2m").subtract(273.15);
      var dewpoint = image.select("dewpoint_temperature_2m").subtract(273.15);
      return ee
        .Image(100)
        .multiply(ee.Image(112).subtract(temp.subtract(dewpoint).multiply(5)))
        .exp()
        .divide(100)
        .rename("Relative_Humidity");
    })
    .median()
    .clip(aoi);

  // 5. Elevation (static)
  var elevationImage = ee
    .Image("USGS/SRTMGL1_003")
    .select("elevation")
    .clip(aoi);

  // 6. Windspeed (ERA5-Land)
  var uWindImage = ee
    .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(aoi)
    .filterDate(start, end)
    .select("u_component_of_wind_10m")
    .median()
    .clip(aoi);
  var vWindImage = ee
    .ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(aoi)
    .filterDate(start, end)
    .select("v_component_of_wind_10m")
    .median()
    .clip(aoi);

  // 7. Precipitation (CHIRPS)
  var precipitationImage = ee
    .ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filterBounds(aoi)
    .filterDate(start, end)
    .select("precipitation")
    .reduce(ee.Reducer.sum())
    .clip(aoi);

  // Combine all bands into a single image
  var combinedImage = hlsImage
    .addBands(soilMoistureImage.rename("sm_surface"))
    .addBands(lstImage.rename("LST_Day_1km"))
    .addBands(humidityImage.rename("Relative_Humidity"))
    .addBands(uWindImage.rename("uWindSpeed_10m"))
    .addBands(vWindImage.rename("vWindSpeed_10m"))
    .addBands(precipitationImage.rename("precipitation"))
    .addBands(elevationImage.rename("elevation"));

  return combinedImage;
}

// Get the combined environmental image for the first report's timestamp
var firstReportImage = getEnvironmentDataImageForTimestamp(firstReport);

// Ensure all bands are of type Float32
var labeledDataWithLocation = firstReportImage.toFloat();

// Export the image as a GeoTIFF to Google Drive
Export.image.toDrive({
  image: labeledDataWithLocation,
  description: "locust_training_data_first_timestamp",
  scale: 1000, // 1Km resolution
  region: aoi,
  maxPixels: 1e13,
  crs: "EPSG:4326",
  folder: "Thesis",
});
