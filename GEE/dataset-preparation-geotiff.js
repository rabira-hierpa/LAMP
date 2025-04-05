var aoi = ee.FeatureCollection(etBoundary);

// Define common projection and scale
var commonScale = 250;
var commonProjection = "EPSG:4326";

// Function to calculate VHI
function calculateVHI(image) {
  var vhi30 = image
    .select("NDVI_30")
    .multiply(0.5)
    .add(image.select("TCI_30").multiply(0.5))
    .rename("VHI_30");
  var vhi60 = image
    .select("NDVI_60")
    .multiply(0.5)
    .add(image.select("TCI_60").multiply(0.5))
    .rename("VHI_60");
  var vhi90 = image
    .select("NDVI_90")
    .multiply(0.5)
    .add(image.select("TCI_90").multiply(0.5))
    .rename("VHI_90");
  return image.addBands([vhi30, vhi60, vhi90]);
}

// Function to calculate TCI
function calculateTCI(image) {
  var tci30 = image
    .select("LST_Day_1km_30")
    .subtract(273.15)
    .multiply(0.1)
    .rename("TCI_30");
  var tci60 = image
    .select("LST_Day_1km_60")
    .subtract(273.15)
    .multiply(0.1)
    .rename("TCI_60");
  var tci90 = image
    .select("LST_Day_1km_90")
    .subtract(273.15)
    .multiply(0.1)
    .rename("TCI_90");
  return image.addBands([tci30, tci60, tci90]);
}

// Function to calculate TVDI
function calculateTVDI(image, geometry) {
  // Define a function to compute TVDI
  var computeTVDI = function (ndvi, lst, ndviTag, lstTag) {
    // Get the LST min and max for the region
    var lstMin = lst
      .reduceRegion({
        reducer: ee.Reducer.min(),
        geometry: geometry,
        scale: 1000,
        maxPixels: 1e9,
      })
      .get(lstTag);

    // Define parameters for the dry edge (simplified approximation)
    var a = 273.15; // Intercept
    var b = 50; // Slope

    // Calculate TVDI
    return lst
      .subtract(lstMin)
      .divide(ee.Image.constant(a).add(ndvi.multiply(b)).subtract(lstMin))
      .rename("TVDI_" + ndviTag.split("_").pop());
  };

  // Calculate TVDI for each time period
  var tvdi30 = computeTVDI(
    image.select("NDVI_30"),
    image.select("LST_Day_1km_30"),
    "NDVI_30",
    "LST_Day_1km_30"
  );

  var tvdi60 = computeTVDI(
    image.select("NDVI_60"),
    image.select("LST_Day_1km_60"),
    "NDVI_60",
    "LST_Day_1km_60"
  );

  var tvdi90 = computeTVDI(
    image.select("NDVI_90"),
    image.select("LST_Day_1km_90"),
    "NDVI_90",
    "LST_Day_1km_90"
  );

  return image.addBands([tvdi30, tvdi60, tvdi90]);
}

// Function to create 7x7x2 image representation for locust reports
function createLocustReportImage(point, date, spatialResolution) {
  var geometry = point.geometry();
  var centerLat = geometry.coordinates().get(1);
  var centerLon = geometry.coordinates().get(0);

  // Create a 7x7 grid centered on the current location
  var gridSize = 7;
  var halfGridSize = Math.floor(gridSize / 2);
  var gridCells = [];

  // Calculate the step size in degrees based on the spatial resolution
  var stepSize = spatialResolution / 111000; // Convert meters to approximate degrees

  // Create a feature collection of grid cells
  for (var i = -halfGridSize; i <= halfGridSize; i++) {
    for (var j = -halfGridSize; j <= halfGridSize; j++) {
      var cellLon = ee.Number(centerLon).add(ee.Number(j).multiply(stepSize));
      var cellLat = ee.Number(centerLat).add(ee.Number(i).multiply(stepSize));

      var cell = ee.Feature(
        ee.Geometry.Rectangle([
          cellLon.subtract(stepSize / 2),
          cellLat.subtract(stepSize / 2),
          cellLon.add(stepSize / 2),
          cellLat.add(stepSize / 2),
        ]),
        {
          row: i + halfGridSize,
          col: j + halfGridSize,
          center_lon: cellLon,
          center_lat: cellLat,
        }
      );

      gridCells.push(cell);
    }
  }

  var grid = ee.FeatureCollection(gridCells);

  // Get previous dates for time-series analysis
  var prevDate30 = date.advance(-30, "days");
  var prevDate60 = date.advance(-60, "days");
  var prevDate90 = date.advance(-90, "days");

  // Filter locust reports for each time period
  var locustReports = ee.FeatureCollection(
    "projects/desert-locust-forcast/assets/FAO_ALL_Reports"
  );

  // Count presence and absence reports for each grid cell for each time period
  function countReportsForPeriod(startDate, endDate) {
    var periodReports = locustReports.filterDate(startDate, endDate);

    var presenceReports = periodReports.filter(
      ee.Filter.eq("Locust Presence", "PRESENT")
    );

    var absenceReports = periodReports.filter(
      ee.Filter.eq("Locust Presence", "ABSENT")
    );

    // Count reports in each grid cell
    var presenceCounts = grid.map(function (cell) {
      var cellGeom = cell.geometry();
      var count = presenceReports.filterBounds(cellGeom).size();
      return cell.set("presence_count", count);
    });

    var absenceCounts = grid.map(function (cell) {
      var cellGeom = cell.geometry();
      var count = absenceReports.filterBounds(cellGeom).size();
      return cell.set("absence_count", count);
    });

    return {
      presence: presenceCounts,
      absence: absenceCounts,
    };
  }

  // Get counts for each time period
  var counts30 = countReportsForPeriod(prevDate30, date);
  var counts60 = countReportsForPeriod(prevDate60, prevDate30);
  var counts90 = countReportsForPeriod(prevDate90, prevDate60);

  // Create image representation (7x7x2) for each time period
  function createPeriodImage(counts, suffix) {
    // Create presence image
    var presenceImg = ee
      .Image()
      .float()
      .paint({
        featureCollection: counts.presence,
        color: "presence_count",
      })
      .rename("presence_" + suffix);

    // Create absence image
    var absenceImg = ee
      .Image()
      .float()
      .paint({
        featureCollection: counts.absence,
        color: "absence_count",
      })
      .rename("absence_" + suffix);

    return ee.Image.cat([presenceImg, absenceImg]);
  }

  var image30 = createPeriodImage(counts30, "30");
  var image60 = createPeriodImage(counts60, "60");
  var image90 = createPeriodImage(counts90, "90");

  return ee.Image.cat([image30, image60, image90]);
}

// Function to extract time-lagged environmental data
function extractTimeLaggedData(point) {
  // Use the parsed date we stored earlier
  var date = point.get("parsed_date")
    ? ee.Date(point.get("parsed_date"))
    : ee.Date(Date.now());

  var geometry = point.geometry();

  var lags = {
    30: date.advance(-30, "days"),
    60: date.advance(-60, "days"),
    90: date.advance(-90, "days"),
  };

  function computeVariable(collectionId, bands, reducer, lag) {
    var reducerFn = reducer === "mean" ? ee.Reducer.mean() : ee.Reducer.sum();
    return ee
      .ImageCollection(collectionId)
      .filterBounds(geometry)
      .filterDate(lags[lag], date)
      .select(bands)
      .reduce(reducerFn)
      .rename(bands[0] + "_" + lag);
  }

  // Extract soil sand content at different depths
  var sandContent0_5 = ee
    .Image("projects/soilgrids-isric/sand_mean")
    .select("sand_0-5cm_mean")
    .rename("sand_0_5cm");

  var sandContent5_15 = ee
    .Image("projects/soilgrids-isric/sand_mean")
    .select("sand_5-15cm_mean")
    .rename("sand_5_15cm");

  var sandContent15_30 = ee
    .Image("projects/soilgrids-isric/sand_mean")
    .select("sand_15-30cm_mean")
    .rename("sand_15_30cm");

  // Add elevation data (SRTM)
  var elevation = ee.Image("USGS/SRTMGL1_003").rename("elevation");

  // Add land cover
  var landcover = ee
    .Image("MODIS/006/MCD12Q1/2019_01_01")
    .select("LC_Type1")
    .rename("landcover");

  // Extract Actual Evapotranspiration data
  var aet30 = computeVariable("MODIS/006/MOD16A2", ["ET"], "sum", "30")
    .multiply(0.1)
    .rename("AET_30");

  var aet60 = computeVariable("MODIS/006/MOD16A2", ["ET"], "sum", "60")
    .multiply(0.1)
    .rename("AET_60");

  var aet90 = computeVariable("MODIS/006/MOD16A2", ["ET"], "sum", "90")
    .multiply(0.1)
    .rename("AET_90");

  // Alternative AET source: TerraClimate monthly product
  var terraClimateAet30 = computeVariable(
    "IDAHO_EPSCOR/TERRACLIMATE",
    ["aet"],
    "sum",
    "30"
  ).rename("TerraClimate_AET_30");

  var terraClimateAet60 = computeVariable(
    "IDAHO_EPSCOR/TERRACLIMATE",
    ["aet"],
    "sum",
    "60"
  ).rename("TerraClimate_AET_60");

  var terraClimateAet90 = computeVariable(
    "IDAHO_EPSCOR/TERRACLIMATE",
    ["aet"],
    "sum",
    "90"
  ).rename("TerraClimate_AET_90");

  // Add Total Biomass Productivity (from TerraClimate)
  var tbp30 = computeVariable(
    "IDAHO_EPSCOR/TERRACLIMATE",
    ["pet"],
    "mean",
    "30"
  ).rename("TBP_30");

  var tbp60 = computeVariable(
    "IDAHO_EPSCOR/TERRACLIMATE",
    ["pet"],
    "mean",
    "60"
  ).rename("TBP_60");

  var tbp90 = computeVariable(
    "IDAHO_EPSCOR/TERRACLIMATE",
    ["pet"],
    "mean",
    "90"
  ).rename("TBP_90");

  var images = [
    computeVariable("MODIS/061/MOD13A2", ["NDVI"], "mean", "30"),
    computeVariable("MODIS/061/MOD13A2", ["NDVI"], "mean", "60"),
    computeVariable("MODIS/061/MOD13A2", ["NDVI"], "mean", "90"),
    computeVariable("MODIS/061/MOD13A2", ["EVI"], "mean", "30"),
    computeVariable("MODIS/061/MOD13A2", ["EVI"], "mean", "60"),
    computeVariable("MODIS/061/MOD13A2", ["EVI"], "mean", "90"),
    computeVariable("MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "30"),
    computeVariable("MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "60"),
    computeVariable("MODIS/061/MOD11A2", ["LST_Day_1km"], "mean", "90"),
    computeVariable("UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "30"),
    computeVariable("UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "60"),
    computeVariable("UCSB-CHG/CHIRPS/DAILY", ["precipitation"], "sum", "90"),
    // Wind at 10m
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["u_component_of_wind_10m"],
      "mean",
      "30"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["u_component_of_wind_10m"],
      "mean",
      "60"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["u_component_of_wind_10m"],
      "mean",
      "90"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["v_component_of_wind_10m"],
      "mean",
      "30"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["v_component_of_wind_10m"],
      "mean",
      "60"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["v_component_of_wind_10m"],
      "mean",
      "90"
    ),
    // Wind at 50m
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["u_component_of_wind_50m"],
      "mean",
      "30"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["u_component_of_wind_50m"],
      "mean",
      "60"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["u_component_of_wind_50m"],
      "mean",
      "90"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["v_component_of_wind_50m"],
      "mean",
      "30"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["v_component_of_wind_50m"],
      "mean",
      "60"
    ),
    computeVariable(
      "ECMWF/ERA5/DAILY",
      ["v_component_of_wind_50m"],
      "mean",
      "90"
    ),
    // Soil Moisture
    computeVariable("NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "30"),
    computeVariable("NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "60"),
    computeVariable("NASA/SMAP/SPL4SMGP/007", ["sm_surface"], "mean", "90"),
    // Add the static data
    elevation,
    landcover,
    // Add the sand content layers
    sandContent0_5,
    sandContent5_15,
    sandContent15_30,
    // Add TBP data
    tbp30,
    tbp60,
    tbp90,
    // Add AET data
    aet30,
    aet60,
    aet90,
    terraClimateAet30,
    terraClimateAet60,
    terraClimateAet90,
  ];

  var combinedImage = ee.Image.cat(images);
  combinedImage = calculateTCI(combinedImage);
  combinedImage = calculateVHI(combinedImage);
  combinedImage = calculateTVDI(combinedImage, geometry);

  // Create 7x7x2 locust report image
  var locustReportImage = createLocustReportImage(point, date, 250); // 250m resolution

  // Combine all data
  return ee.Image.cat([combinedImage, locustReportImage]);
}

// Function to create an export task
function createExportTask(featureIndex, feature) {
  // Get the observation date as a string
  var obsDateStr = feature.get("Obs Date");

  var obsDate;
  var formattedDate;
  var obsDateClient = null;

  try {
    // Convert to JavaScript string properly

    if (obsDateStr) {
      obsDateClient = String(obsDateStr.getInfo()); // Explicitly convert to JS String
    }

    // Check if we have a valid date string
    if (obsDateClient && obsDateClient.indexOf("/") !== -1) {
      // Use indexOf instead of includes
      // Try MM/DD/YYYY format
      var dateParts = obsDateClient.split(" ")[0].split("/");

      // Check if we have enough parts and its after the year 2000
      if (dateParts.length >= 3 && dateParts[2] > 2000) {
        var month = dateParts[0];
        var day = dateParts[1];
        var year = dateParts[2];
        // Make sure month and day are two digits
        month = month.length === 1 ? "0" + month : month;
        day = day.length === 1 ? "0" + day : day;

        formattedDate = year + "-" + month + "-" + day;
        obsDate = ee.Date(formattedDate);
        print("Parsed date:", formattedDate);
      } else {
        throw new Error(
          "Invalid date format: not enough parts after splitting"
        );
      }
    } else {
      throw new Error("Invalid or empty date format");
    }
  } catch (e) {
    console.error("Error parsing date:", e);
    // Use current date as fallback
    var now = new Date();
    var year = now.getFullYear();
    var month = String(now.getMonth() + 1).padStart(2, "0");
    var day = String(now.getDate()).padStart(2, "0");
    formattedDate = year + "-" + month + "-" + day;
    obsDate = ee.Date(formattedDate);
    print("Using fallback date:", formattedDate);
  }

  // Get the locust presence value
  var presence = feature.get("Locust Presence").getInfo();

  // Make a copy of the feature with the parsed date
  var featureWithParsedDate = feature.set("parsed_date", obsDate);

  // Extract time-lagged data using the feature with parsed date
  var timeLaggedData = extractTimeLaggedData(featureWithParsedDate).toFloat();
  var patchGeometry = feature.geometry().buffer(10000);

  var multiBandImage = ee.Image.cat([
    timeLaggedData,
    ee.Image.constant(presence === "PRESENT" ? 1 : 0)
      .toFloat()
      .rename("label"),
  ]).clip(patchGeometry);

  // Construct a descriptive name for the export task
  var exportDescription =
    "locust_" +
    formattedDate +
    "_label_" +
    (presence === "PRESENT" ? "1" : "0") +
    "_" +
    (featureIndex + 1);

  // Create the export task
  Export.image.toDrive({
    image: multiBandImage,
    description: exportDescription,
    scale: commonScale,
    region: patchGeometry,
    maxPixels: 1e13,
    crs: commonProjection,
    folder: "Locust_Export",
  });
}

// Load FAO locust data
var faoReportAssetId = "projects/desert-locust-forcast/assets/FAO_ALL_Reports";
var locustData = ee.FeatureCollection(faoReportAssetId);

var presencePoints = locustData.filter(
  ee.Filter.eq("Locust Presence", "PRESENT")
);
var absencePoints = locustData.filter(
  ee.Filter.eq("Locust Presence", "ABSENT")
);

// Print sample points for inspection
print("Presence points sample:", presencePoints.size());
print("Absence points sample:", absencePoints.size());

// Get a single presence point for testing
var singlePoint = presencePoints.filter(ee.Filter.eq("index", 0));
print("Single test point:", singlePoint);

// Process this single point
Map.centerObject(aoi, 6);
Map.addLayer(
  singlePoint.geometry(),
  { color: "red", radius: 10 },
  "Locust presence point"
);

// Manually call the export task function for testing
createExportTask(0, singlePoint);

// If the test is successful, uncomment the following to process more points:
/*
// Export presence points
presencePoints.evaluate(function(features) {
    if (features && features.features) {
        for (var i = 0; i < features.features.length; i++) {
            var feature = ee.Feature(features.features[i]);
            createExportTask(i, feature);
        }
    } else {
        print('No presence features found or error in data structure');
    }
});

// Export absence points
absencePoints.evaluate(function(features) {
    if (features && features.features) {
        for (var i = 0; i < features.features.length; i++) {
            var feature = ee.Feature(features.features[i]);
            createExportTask(i + sampleSize, feature);
        }
    } else {
        print('No absence features found or error in data structure');
    }
});
*/
