/**
 * Desert locust presence prediction using Multi-Scale Approach
 * Based on Klein et al. (2022) methodology for multi-scale analysis of locust breeding grounds
 */

// Define Ethiopia boundary - Needs to be set before running
var etBoundary = ee
  .FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq("country_na", "Ethiopia"));

var aoi = etBoundary;

// Define multiple spatial scales for analysis
var scales = {
  coarse: 1000, // 1km resolution for broad climate patterns
  medium: 250, // 250m resolution for MODIS-based analysis
  fine: 30, // 30m resolution for Landsat/Sentinel-based detailed analysis
};

// Define common projection
var commonProjection = "EPSG:4326";

// Global tracker for missing data
var missingDataVariables = [];

// Calculate PLAN indices as described in the Tabar et al. paper
function calculatePLANIndices(image) {
  // Vegetation Health Index (VHI) - 0.5*NDVI + 0.5*TCI
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

  // Normalized Difference Water Index (NDWI) using NIR and SWIR
  // Note: For MODIS, using bands 2 (NIR) and 6 (SWIR)
  var ndwi30 = image.select("NDWI_30").rename("NDWI_30");
  var ndwi60 = image.select("NDWI_60").rename("NDWI_60");
  var ndwi90 = image.select("NDWI_90").rename("NDWI_90");

  // Temperature Vegetation Dryness Index (TVDI) - (LST - LSTmin)/(a + b*NDVI - LSTmin)
  // Simplified approach as per PLAN paper
  var tvdi30 = image
    .select("LST_Day_1km_30")
    .subtract(273.15)
    .divide(image.select("NDVI_30").multiply(10).add(273.15))
    .rename("TVDI_30");
  var tvdi60 = image
    .select("LST_Day_1km_60")
    .subtract(273.15)
    .divide(image.select("NDVI_60").multiply(10).add(273.15))
    .rename("TVDI_60");
  var tvdi90 = image
    .select("LST_Day_1km_90")
    .subtract(273.15)
    .divide(image.select("NDVI_90").multiply(10).add(273.15))
    .rename("TVDI_90");

  return image.addBands([
    vhi30,
    vhi60,
    vhi90,
    ndwi30,
    ndwi60,
    ndwi90,
    tvdi30,
    tvdi60,
    tvdi90,
  ]);
}

// Function to calculate Temperature Condition Index (TCI) as described in PLAN
function calculateTCI(image) {
  // Convert from Kelvin to Celsius and scale
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

// Function to create context window for locust reports - 7x7 grid
function createLocustContextWindow(point, date, spatialResolution) {
  var geometry = point.geometry();
  var centerLat = geometry.coordinates().get(1);
  var centerLon = geometry.coordinates().get(0);

  // PLAN paper uses 7x7 grid cells (49 total cells)
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

  // Get previous dates for time-series analysis (t-30, t-60, t-90)
  var prevDate30 = date.advance(-30, "days");
  var prevDate60 = date.advance(-60, "days");
  var prevDate90 = date.advance(-90, "days");

  // Filter locust reports for each time period
  var locustReports = ee.FeatureCollection(
    "projects/desert-locust-forcast/assets/FAO_filtered_data_2000"
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

// Function to extract medium-scale time-lagged environmental data using MODIS
function extractPLANVariables(point) {
  // Reset missing data tracker for this point
  missingDataVariables = [];

  // Use the parsed date we stored earlier
  var date = ee.Date(point.get("parsed_date") || Date.now());
  var geometry = point.geometry();

  // Time lags as specified in PLAN (30, 60, 90 days)
  var lags = {
    30: date.advance(-30, "days"),
    60: date.advance(-60, "days"),
    90: date.advance(-90, "days"),
  };

  // Function to safely compute variables from collections
  function computeVariable(collectionId, bands, reducer, lag) {
    try {
      var reducerFn = reducer === "mean" ? ee.Reducer.mean() : ee.Reducer.sum();

      var collection = ee
        .ImageCollection(collectionId)
        .filterBounds(geometry)
        .filterDate(lags[lag], date);

      // Check if collection is empty
      var size = collection.size().getInfo();
      if (size === 0) {
        var variableName = collectionId + " " + bands[0] + " " + lag;
        print("Warning: Empty collection for", collectionId, bands[0], lag);
        missingDataVariables.push(variableName);
        return ee.Image(0).rename(bands[0] + "_" + lag);
      }

      // Process the collection
      return collection
        .select(bands)
        .reduce(reducerFn)
        .rename(bands[0] + "_" + lag);
    } catch (e) {
      print("Error in computeVariable:", e);
      var variableName = collectionId + " " + bands[0] + " " + lag;
      missingDataVariables.push(variableName);
      return ee.Image(0).rename(bands[0] + "_" + lag);
    }
  }

  // Calculate NDWI using alternative method with MOD09GA
  function computeNDWI(lag) {
    try {
      // First try to use MOD09GA
      var collection = ee
        .ImageCollection("MODIS/006/MOD09GA")
        .filterBounds(geometry)
        .filterDate(lags[lag], date);

      var size = collection.size().getInfo();
      if (size === 0) {
        print("Warning: Empty collection for NDWI", lag);
        print("Using approximated NDWI from MOD13Q1 (NDVI and EVI)");

        // Alternative: Approximate NDWI using NDVI and EVI from MOD13Q1
        var ndvi = computeVariable("MODIS/061/MOD13Q1", ["NDVI"], "mean", lag);
        var evi = computeVariable("MODIS/061/MOD13Q1", ["EVI"], "mean", lag);

        // Create a simple water index proxy using relationship between NDVI and EVI
        return evi
          .subtract(ndvi)
          .multiply(0.5)
          .add(0.5)
          .rename("NDWI_" + lag);
      }

      var ndwiCollection = collection.map(function (img) {
        // NDWI = (NIR - SWIR) / (NIR + SWIR)
        // For MODIS, using bands 2 (NIR) and 6 (SWIR)
        return img
          .normalizedDifference(["sur_refl_b02", "sur_refl_b06"])
          .rename("NDWI");
      });

      return ndwiCollection.mean().rename("NDWI_" + lag);
    } catch (e) {
      print("Error computing NDWI:", e);
      missingDataVariables.push("NDWI_" + lag);
      return ee.Image(0).rename("NDWI_" + lag);
    }
  }

  // Extract static environmental data
  // Soil parameters
  var soilParameters = ee
    .Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
    .select("b0")
    .rename("soil_texture");

  // Elevation from SRTM
  var elevation = ee.Image("USGS/SRTMGL1_003").rename("elevation");

  // Derive slope and aspect
  var terrain = ee.Algorithms.Terrain(elevation);
  var slope = terrain.select("slope").rename("slope");
  var aspect = terrain.select("aspect").rename("aspect");

  // Land cover (MODIS MCD12Q1)
  var landcover = ee
    .Image("MODIS/006/MCD12Q1/2019_01_01")
    .select("LC_Type1")
    .rename("landcover");

  // Calculate dynamic variables
  var ndvi30 = computeVariable("MODIS/061/MOD13Q1", ["NDVI"], "mean", "30");
  var ndvi60 = computeVariable("MODIS/061/MOD13Q1", ["NDVI"], "mean", "60");
  var ndvi90 = computeVariable("MODIS/061/MOD13Q1", ["NDVI"], "mean", "90");

  var evi30 = computeVariable("MODIS/061/MOD13Q1", ["EVI"], "mean", "30");
  var evi60 = computeVariable("MODIS/061/MOD13Q1", ["EVI"], "mean", "60");
  var evi90 = computeVariable("MODIS/061/MOD13Q1", ["EVI"], "mean", "90");

  var lst30 = computeVariable(
    "MODIS/061/MOD11A2",
    ["LST_Day_1km"],
    "mean",
    "30"
  );
  var lst60 = computeVariable(
    "MODIS/061/MOD11A2",
    ["LST_Day_1km"],
    "mean",
    "60"
  );
  var lst90 = computeVariable(
    "MODIS/061/MOD11A2",
    ["LST_Day_1km"],
    "mean",
    "90"
  );

  var precip30 = computeVariable(
    "UCSB-CHG/CHIRPS/DAILY",
    ["precipitation"],
    "sum",
    "30"
  );
  var precip60 = computeVariable(
    "UCSB-CHG/CHIRPS/DAILY",
    ["precipitation"],
    "sum",
    "60"
  );
  var precip90 = computeVariable(
    "UCSB-CHG/CHIRPS/DAILY",
    ["precipitation"],
    "sum",
    "90"
  );

  // Wind components - Use alternative source NCEP/NCAR Reanalysis if ERA5 is not available
  function computeWindComponents(lag) {
    try {
      // Try ERA5 first
      var era5Collection = ee
        .ImageCollection("ECMWF/ERA5/DAILY")
        .filterBounds(geometry)
        .filterDate(lags[lag], date);

      var size = era5Collection.size().getInfo();
      if (size === 0) {
        print(
          "Warning: ERA5 wind data not available, using NCEP/NCAR Reanalysis for lag",
          lag
        );

        // Alternative source: NCEP/NCAR Reanalysis
        var ncepCollection = ee
          .ImageCollection("NCEP_DOE_II/daily_averages")
          .filterBounds(geometry)
          .filterDate(lags[lag], date);

        // Check if NCEP data is available
        var ncepSize = ncepCollection.size().getInfo();
        if (ncepSize === 0) {
          print("Warning: No wind data available for lag", lag);
          missingDataVariables.push("Wind_" + lag);
          return {
            u: ee.Image(0).rename("u_component_of_wind_10m_" + lag),
            v: ee.Image(0).rename("v_component_of_wind_10m_" + lag),
          };
        }

        // Process NCEP data
        var uWind = ncepCollection
          .select(["uwnd_10m"])
          .mean()
          .rename("u_component_of_wind_10m_" + lag);
        var vWind = ncepCollection
          .select(["vwnd_10m"])
          .mean()
          .rename("v_component_of_wind_10m_" + lag);

        return {
          u: uWind,
          v: vWind,
        };
      }

      // Process ERA5 data
      var uWind = era5Collection
        .select(["u_component_of_wind_10m"])
        .mean()
        .rename("u_component_of_wind_10m_" + lag);
      var vWind = era5Collection
        .select(["v_component_of_wind_10m"])
        .mean()
        .rename("v_component_of_wind_10m_" + lag);

      return {
        u: uWind,
        v: vWind,
      };
    } catch (e) {
      print("Error computing wind components:", e);
      missingDataVariables.push("Wind_" + lag);
      return {
        u: ee.Image(0).rename("u_component_of_wind_10m_" + lag),
        v: ee.Image(0).rename("v_component_of_wind_10m_" + lag),
      };
    }
  }

  // Compute wind components for each lag period
  var wind30 = computeWindComponents("30");
  var wind60 = computeWindComponents("60");
  var wind90 = computeWindComponents("90");

  // Soil moisture (SMAP)
  var soilMoisture30 = computeVariable(
    "NASA/SMAP/SPL4SMGP/007",
    ["sm_surface"],
    "mean",
    "30"
  );
  var soilMoisture60 = computeVariable(
    "NASA/SMAP/SPL4SMGP/007",
    ["sm_surface"],
    "mean",
    "60"
  );
  var soilMoisture90 = computeVariable(
    "NASA/SMAP/SPL4SMGP/007",
    ["sm_surface"],
    "mean",
    "90"
  );

  // NDWI calculation
  var ndwi30 = computeNDWI("30");
  var ndwi60 = computeNDWI("60");
  var ndwi90 = computeNDWI("90");

  // Combine all variables as a single multi-band image
  var dynamicVariables = ee.Image.cat([
    ndvi30,
    ndvi60,
    ndvi90,
    evi30,
    evi60,
    evi90,
    lst30,
    lst60,
    lst90,
    precip30,
    precip60,
    precip90,
    wind30.u,
    wind60.u,
    wind90.u,
    wind30.v,
    wind60.v,
    wind90.v,
    soilMoisture30,
    soilMoisture60,
    soilMoisture90,
    ndwi30,
    ndwi60,
    ndwi90,
  ]);

  // Combine with static variables
  var allVariables = ee.Image.cat([
    dynamicVariables,
    elevation,
    slope,
    aspect,
    landcover,
    soilParameters,
  ]);

  // Calculate derived indices (TCI, VHI, TVDI)
  allVariables = calculateTCI(allVariables);
  allVariables = calculatePLANIndices(allVariables);

  return allVariables;
}

// Function to extract coarse-scale climate patterns (Klein approach)
function extractCoarseScaleFeatures(geometry, date) {
  // Time lags - longer for climate patterns
  var lags = {
    30: date.advance(-30, "days"),
    90: date.advance(-90, "days"),
    180: date.advance(-180, "days"), // 6-month climate context
  };

  // ERA5 climate reanalysis (coarse resolution but comprehensive)
  try {
    var era5_30 = ee
      .ImageCollection("ECMWF/ERA5/DAILY")
      .filterBounds(geometry)
      .filterDate(lags["30"], date)
      .select([
        "mean_2m_air_temperature",
        "total_precipitation",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
      ])
      .mean()
      .rename([
        "temp_era5_30",
        "precip_era5_30",
        "u_wind_era5_30",
        "v_wind_era5_30",
      ]);
  } catch (e) {
    print("Error getting ERA5 data (30 day):", e);
    missingDataVariables.push("ERA5_30");
    var era5_30 = ee
      .Image([0, 0, 0, 0])
      .rename([
        "temp_era5_30",
        "precip_era5_30",
        "u_wind_era5_30",
        "v_wind_era5_30",
      ]);
  }

  try {
    var era5_90 = ee
      .ImageCollection("ECMWF/ERA5/DAILY")
      .filterBounds(geometry)
      .filterDate(lags["90"], date)
      .select([
        "mean_2m_air_temperature",
        "total_precipitation",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
      ])
      .mean()
      .rename([
        "temp_era5_90",
        "precip_era5_90",
        "u_wind_era5_90",
        "v_wind_era5_90",
      ]);
  } catch (e) {
    print("Error getting ERA5 data (90 day):", e);
    missingDataVariables.push("ERA5_90");
    var era5_90 = ee
      .Image([0, 0, 0, 0])
      .rename([
        "temp_era5_90",
        "precip_era5_90",
        "u_wind_era5_90",
        "v_wind_era5_90",
      ]);
  }

  // Add seasonal climate oscillation indices
  // Get day of year and compute seasonal cycle components (Klein approach)
  var dayOfYear = date.getRelative("day", "year");
  var seasonalCycle = ee
    .Image([
      ee
        .Image(dayOfYear)
        .multiply((2 * Math.PI) / 365)
        .cos(), // Cosine component
      ee
        .Image(dayOfYear)
        .multiply((2 * Math.PI) / 365)
        .sin(), // Sine component
    ])
    .rename(["seasonal_cos", "seasonal_sin"]);

  // Klein uses climate zones - get WWF biome data
  var ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017");
  var pointEcoregion = ecoregions.filterBounds(geometry).first();

  // Extract climate zone (biome) as a regional indicator (Klein approach)
  var biomeId = ee.Number.parse(pointEcoregion.get("BIOME_NUM")).toInt();
  var biomeImage = ee.Image.constant(biomeId).toInt().rename("biome_id");

  // Get long-term climate means from WorldClim
  var worldClim = ee
    .Image("WORLDCLIM/V1/MONTHLY")
    .select(["tavg01", "tavg07", "prec01", "prec07"])
    .rename([
      "temp_jan_mean",
      "temp_jul_mean",
      "prec_jan_mean",
      "prec_jul_mean",
    ]);

  return ee.Image.cat([era5_30, era5_90, seasonalCycle, biomeImage, worldClim]);
}

// Function to calculate texture metrics (GLCM) following Klein's approach
function calculateTextureMetrics(image, bandName) {
  // Generate GLCM texture features
  var glcm = image.select(bandName).glcmTexture({ size: 3 });

  // Select key texture features as used by Klein
  return glcm
    .select(["contrast", "correlation", "ent", "var"])
    .rename([
      "texture_contrast",
      "texture_correlation",
      "texture_entropy",
      "texture_variance",
    ]);
}

// Calculate Shannon diversity index from land cover frequency histogram
function calculateShannonDiversity(histogram) {
  var dict = ee.Dictionary(histogram);
  var total = dict.values().reduce(ee.Reducer.sum());

  var shannon = dict
    .toArray()
    .map(function (count) {
      var proportion = ee.Number(count).divide(total);
      return proportion.multiply(proportion.log());
    })
    .reduce(ee.Reducer.sum(), [0])
    .multiply(-1);

  return shannon;
}

// Extract landscape metrics as done in Klein et al.
function extractLandscapeMetrics(geometry, scale) {
  // Use recent land cover data
  var landcover = ee
    .Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
    .select("discrete_classification")
    .clip(geometry.buffer(5000));

  // Calculate landscape metrics
  var classes = landcover.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: geometry.buffer(2000),
    scale: scale,
    maxPixels: 1e9,
  });

  // Convert histogram to landscape metrics
  // This is a simplified version of Klein's approach
  var patchDensity = ee
    .Number(ee.Dictionary(classes.get("discrete_classification")).size())
    .divide(geometry.buffer(2000).area())
    .multiply(1000000); // patches per km²

  var shannonDiversity = calculateShannonDiversity(
    classes.get("discrete_classification")
  );

  return ee
    .Image([patchDensity, shannonDiversity])
    .rename(["patch_density", "shannon_diversity"]);
}

// Function to extract high-resolution data from Sentinel-2 or Landsat (Klein approach)
function extractFineScaleFeatures(geometry, date) {
  // Time window for Sentinel-2 search (narrower due to frequent revisits)
  var startDate = date.advance(-30, "days");
  var endDate = date;

  try {
    // Load Sentinel-2 imagery
    var s2Collection = ee
      .ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(geometry)
      .filterDate(startDate, endDate)
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)); // Filter cloudy images

    // If no clear Sentinel-2 images, try Landsat
    var s2Size = s2Collection.size().getInfo();

    if (s2Size === 0) {
      print("No clear Sentinel-2 images, trying Landsat");
      var landsat = ee
        .ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(geometry)
        .filterDate(startDate, endDate);

      var landsatSize = landsat.size().getInfo();

      if (landsatSize === 0) {
        print("No Landsat images available either");
        missingDataVariables.push("Fine_Scale_Imagery");

        // Return placeholder image with zeros
        return ee
          .Image([0, 0, 0, 0, 0, 0, 0, 0])
          .rename([
            "fine_ndvi",
            "fine_ndwi",
            "fine_rededge",
            "texture_contrast",
            "texture_correlation",
            "texture_entropy",
            "texture_variance",
            "patch_density",
            "shannon_diversity",
          ]);
      }

      // Process Landsat data
      var fineScale = landsat.median();

      // Calculate indices
      var ndvi = fineScale
        .normalizedDifference(["SR_B5", "SR_B4"])
        .rename("fine_ndvi");
      var ndwi = fineScale
        .normalizedDifference(["SR_B3", "SR_B5"])
        .rename("fine_ndwi");

      // No red edge for Landsat
      var rededge = ee.Image(0).rename("fine_rededge");

      // Calculate texture
      var texture = calculateTextureMetrics(fineScale, "SR_B5"); // NIR band
    } else {
      // Process Sentinel-2 data
      var fineScale = s2Collection.median();

      // Calculate indices
      var ndvi = fineScale
        .normalizedDifference(["B8", "B4"])
        .rename("fine_ndvi");
      var ndwi = fineScale
        .normalizedDifference(["B3", "B8"])
        .rename("fine_ndwi");
      var rededge = fineScale
        .normalizedDifference(["B8", "B5"])
        .rename("fine_rededge");

      // Calculate texture
      var texture = calculateTextureMetrics(fineScale, "B8"); // NIR band
    }

    // Derive landscape metrics (Klein approach)
    var lcObj = extractLandscapeMetrics(geometry, 30); // 30m scale landscape metrics

    return ee.Image.cat([ndvi, ndwi, rededge, texture, lcObj]);
  } catch (e) {
    print("Error extracting fine scale features:", e);
    missingDataVariables.push("Fine_Scale_Imagery_Error");

    // Return placeholder image with zeros
    return ee
      .Image([0, 0, 0, 0, 0, 0, 0, 0, 0])
      .rename([
        "fine_ndvi",
        "fine_ndwi",
        "fine_rededge",
        "texture_contrast",
        "texture_correlation",
        "texture_entropy",
        "texture_variance",
        "patch_density",
        "shannon_diversity",
      ]);
  }
}

// Function to extract multi-scale environmental variables
function extractMultiScaleVariables(point) {
  var date = ee.Date(point.get("parsed_date") || Date.now());
  var geometry = point.geometry();

  // Reset missing data tracker
  missingDataVariables = [];

  // Extract coarse-scale features (climate, regional patterns)
  var coarseFeatures = extractCoarseScaleFeatures(geometry, date);

  // Extract medium-scale features (MODIS-based features)
  var mediumFeatures = extractPLANVariables(point);

  // Extract fine-scale features (Sentinel-2 or Landsat)
  var fineFeatures = extractFineScaleFeatures(geometry, date);

  // Create 7x7x2 locust context window
  var locustContextImage = createLocustContextWindow(point, date, 250);

  // Combine all scale features
  return ee.Image.cat([
    coarseFeatures,
    mediumFeatures,
    fineFeatures,
    locustContextImage,
  ]);
}

// Function to integrate features from multiple scales following Klein's approach
function integrateMultiScaleFeatures(multiScaleImage, point) {
  // Get point-specific weights based on land cover and seasonality
  var weights = calculateScaleWeights(point);

  // Extract scale-specific feature groups
  var coarseFeatures = multiScaleImage.select([
    "temp_era5_.*",
    "precip_era5_.*",
    "u_wind_era5_.*",
    "v_wind_era5_.*",
    "seasonal_.*",
    "biome_id",
    "temp_.*_mean",
    "prec_.*_mean",
  ]);
  var mediumFeatures = multiScaleImage.select([
    "NDVI_.*",
    "EVI_.*",
    "LST_Day_.*",
    "precipitation_.*",
    "sm_surface_.*",
    "u_component_of_wind_10m_.*",
    "v_component_of_wind_10m_.*",
    "NDWI_.*",
    "VHI_.*",
    "TVDI_.*",
    "TCI_.*",
  ]);
  var fineFeatures = multiScaleImage.select([
    "fine_.*",
    "texture_.*",
    "patch_density",
    "shannon_diversity",
  ]);

  // Apply scale-specific weights
  var weightedCoarse = coarseFeatures.multiply(weights.get("coarse"));
  var weightedMedium = mediumFeatures.multiply(weights.get("medium"));
  var weightedFine = fineFeatures.multiply(weights.get("fine"));

  // Combine weighted features
  return ee.Image.cat([weightedCoarse, weightedMedium, weightedFine]);
}

// Calculate dynamic scale weights based on Klein's methodology
function calculateScaleWeights(point) {
  // Extract contextual information
  var date = ee.Date(point.get("parsed_date") || Date.now());
  var month = date.get("month").getInfo();
  var season = month >= 3 && month <= 8 ? "breeding" : "migration";

  // Try to get landcover class from feature properties
  var landcover;
  try {
    landcover = point.get("landcover_class").getInfo();
  } catch (e) {
    // If not available, extract from MODIS landcover
    var lcImage = ee.Image("MODIS/006/MCD12Q1/2019_01_01").select("LC_Type1");
    landcover = lcImage
      .reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: point.geometry(),
        scale: 500,
      })
      .get("LC_Type1")
      .getInfo();
  }

  // Default weights
  var weights = {
    coarse: 0.3,
    medium: 0.4,
    fine: 0.3,
  };

  // Adjust weights based on season (following Klein's approach)
  if (season === "breeding") {
    weights.fine = 0.5; // Fine scale more important during breeding
    weights.medium = 0.3;
    weights.coarse = 0.2;
  } else if (season === "migration") {
    weights.coarse = 0.5; // Coarse scale more important during migration
    weights.medium = 0.4;
    weights.fine = 0.1;
  }

  // Adjust weights based on land cover (following Klein's findings)
  // MODIS Land cover classes: 12=cropland, 16=barren
  if (landcover === 12) {
    weights.fine = Math.min(weights.fine + 0.2, 0.6); // Fine-scale more important in croplands
  } else if (landcover === 16) {
    weights.coarse = Math.min(weights.coarse + 0.2, 0.6); // Coarse-scale more important in desert/barren
  }

  return ee.Dictionary(weights);
}

// Function to create an export task for a feature with multi-scale approach
function createMultiScaleExportTask(featureIndex, feature) {
  // Make sure feature is an ee.Feature
  if (!feature) {
    print("Error: Null feature at index", featureIndex);
    return;
  }

  // Ensure feature is an ee.Feature
  feature = ee.Feature(feature);

  // Handle date parsing
  var obsDateStr = feature.get("Obs Date");
  var obsDate;
  var formattedDate;
  var obsDateClient = null;

  try {
    if (obsDateStr) {
      obsDateClient = String(obsDateStr.getInfo());
    }

    if (obsDateClient && obsDateClient.indexOf("/") !== -1) {
      var dateParts = obsDateClient.split(" ")[0].split("/");

      if (dateParts.length >= 3 && dateParts[2] >= 2000) {
        var month = dateParts[0];
        var day = dateParts[1];
        var year = dateParts[2];
        month = month.length === 1 ? "0" + month : month;
        day = day.length === 1 ? "0" + day : day;

        formattedDate = year + "-" + month + "-" + day;
        obsDate = ee.Date(formattedDate);
        print("Parsed date:", formattedDate);
      } else {
        throw new Error("Invalid date format");
      }
    } else {
      throw new Error("Invalid date string");
    }
  } catch (e) {
    print("Error parsing date:", e, "Using current date as fallback.");
    var now = new Date();
    var year = now.getFullYear();
    var month = String(now.getMonth() + 1).padStart(2, "0");
    var day = String(now.getDate()).padStart(2, "0");
    formattedDate = year + "-" + month + "-" + day;
    obsDate = ee.Date(formattedDate);
  }

  // Get presence/absence label
  var presence;
  try {
    presence = feature.get("Locust Presence").getInfo();
  } catch (e) {
    print("Error getting locust presence:", e);
    presence = "UNKNOWN";
  }

  // Add parsed date to feature
  var featureWithDate = feature.set("parsed_date", obsDate);

  // Determine season (breeding vs migration) based on month
  // Klein et al. found seasonal differences in habitat preferences
  var month = obsDate.get("month").getInfo();
  var season = month >= 3 && month <= 8 ? "breeding" : "migration";
  featureWithDate = featureWithDate.set("season", season);

  // Get landcover class at the point location
  var landcover = ee
    .Image("MODIS/006/MCD12Q1/2019_01_01")
    .select("LC_Type1")
    .reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: feature.geometry(),
      scale: 500,
    })
    .get("LC_Type1");

  featureWithDate = featureWithDate.set("landcover_class", landcover);

  // Extract multi-scale environmental variables
  var multiScaleData = extractMultiScaleVariables(featureWithDate).toFloat();

  // Integrate features from multiple scales using Klein's approach
  var integratedData = integrateMultiScaleFeatures(
    multiScaleData,
    featureWithDate
  );

  // Calculate habitat suitability index (Klein approach)
  var hsiData = calculateHabitatSuitabilityIndex(multiScaleData);

  // Check if critical data is missing
  if (hasCriticalDataMissing()) {
    print(
      "❌ SKIPPING EXPORT: Critical data missing for feature",
      featureIndex,
      "with date",
      formattedDate
    );
    print("Missing variables:", missingDataVariables);
    return;
  }

  // Create buffer around point (Klein uses variable buffer sizes based on ecology)
  var patchGeometry = feature.geometry().buffer(5000);

  // Add label band (1 for presence, 0 for absence)
  var presenceValue = presence === "PRESENT" ? 1 : 0;

  // Create final image with original data, integrated data, HSI, and label
  var finalImage = ee.Image.cat([
    multiScaleData, // All original multi-scale features
    integratedData, // Scale-integrated features
    hsiData, // Habitat suitability index
    ee.Image.constant(presenceValue).toFloat().rename("label"),
  ]).clip(patchGeometry);

  // Create export task with Klein multi-scale approach naming
  var exportDescription =
    "Klein_MultiScale_locust_" +
    formattedDate +
    "_label_" +
    presenceValue +
    "_idx_" +
    featureIndex;

  Export.image.toDrive({
    image: finalImage,
    description: exportDescription,
    scale: scales.medium, // Use medium scale for export
    region: patchGeometry,
    maxPixels: 1e13,
    crs: commonProjection,
    folder: "MultiScale_Locust_Export",
  });

  print("✅ Created multi-scale export task:", exportDescription);
  print("Season:", season);
  print("Landcover class:", landcover.getInfo());
  print(
    "Non-critical missing variables:",
    missingDataVariables.length > 0 ? missingDataVariables : "None"
  );

  // Visualize multi-scale features if this is the first feature
  if (featureIndex === 0) {
    visualizeMultiScaleFeatures(multiScaleData, patchGeometry);
  }
}

// Function to create an export task for a feature with multi-scale approach
function createMultiScaleExportTask(featureIndex, feature) {
  // Make sure feature is an ee.Feature
  if (!feature) {
    print("Error: Null feature at index", featureIndex);
    return;
  }

  // Ensure feature is an ee.Feature
  feature = ee.Feature(feature);

  // Handle date parsing
  var obsDateStr = feature.get("Obs Date");
  var obsDate;
  var formattedDate;
  var obsDateClient = null;

  try {
    if (obsDateStr) {
      obsDateClient = String(obsDateStr.getInfo());
    }

    if (obsDateClient && obsDateClient.indexOf("/") !== -1) {
      var dateParts = obsDateClient.split(" ")[0].split("/");

      if (dateParts.length >= 3 && dateParts[2] >= 2000) {
        var month = dateParts[0];
        var day = dateParts[1];
        var year = dateParts[2];
        month = month.length === 1 ? "0" + month : month;
        day = day.length === 1 ? "0" + day : day;

        formattedDate = year + "-" + month + "-" + day;
        obsDate = ee.Date(formattedDate);
        print("Parsed date:", formattedDate);
      } else {
        throw new Error("Invalid date format");
      }
    } else {
      throw new Error("Invalid date string");
    }
  } catch (e) {
    print("Error parsing date:", e, "Using current date as fallback.");
    var now = new Date();
    var year = now.getFullYear();
    var month = String(now.getMonth() + 1).padStart(2, "0");
    var day = String(now.getDate()).padStart(2, "0");
    formattedDate = year + "-" + month + "-" + day;
    obsDate = ee.Date(formattedDate);
  }

  // Get presence/absence label
  var presence;
  try {
    presence = feature.get("Locust Presence").getInfo();
  } catch (e) {
    print("Error getting locust presence:", e);
    presence = "UNKNOWN";
  }

  // Add parsed date to feature
  var featureWithDate = feature.set("parsed_date", obsDate);

  // Determine season (breeding vs migration) based on month
  // Klein et al. found seasonal differences in habitat preferences
  var month = obsDate.get("month").getInfo();
  var season = month >= 3 && month <= 8 ? "breeding" : "migration";
  featureWithDate = featureWithDate.set("season", season);

  // Get landcover class at the point location
  var landcover = ee
    .Image("MODIS/006/MCD12Q1/2019_01_01")
    .select("LC_Type1")
    .reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: feature.geometry(),
      scale: 500,
    })
    .get("LC_Type1");

  featureWithDate = featureWithDate.set("landcover_class", landcover);

  // Extract multi-scale environmental variables
  var multiScaleData = extractMultiScaleVariables(featureWithDate).toFloat();

  // Integrate features from multiple scales using Klein's approach
  var integratedData = integrateMultiScaleFeatures(
    multiScaleData,
    featureWithDate
  );

  // Calculate habitat suitability index (Klein approach)
  var hsiData = calculateHabitatSuitabilityIndex(multiScaleData);

  // Check if critical data is missing
  if (hasCriticalDataMissing()) {
    print(
      "❌ SKIPPING EXPORT: Critical data missing for feature",
      featureIndex,
      "with date",
      formattedDate
    );
    print("Missing variables:", missingDataVariables);
    return;
  }

  // Create buffer around point (Klein uses variable buffer sizes based on ecology)
  var patchGeometry = feature.geometry().buffer(5000);

  // Add label band (1 for presence, 0 for absence)
  var presenceValue = presence === "PRESENT" ? 1 : 0;

  // Create final image with original data, integrated data, HSI, and label
  var finalImage = ee.Image.cat([
    multiScaleData, // All original multi-scale features
    integratedData, // Scale-integrated features
    hsiData, // Habitat suitability index
    ee.Image.constant(presenceValue).toFloat().rename("label"),
  ]).clip(patchGeometry);

  // Create export task with Klein multi-scale approach naming
  var exportDescription =
    "Klein_MultiScale_locust_" +
    formattedDate +
    "_label_" +
    presenceValue +
    "_idx_" +
    featureIndex;

  Export.image.toDrive({
    image: finalImage,
    description: exportDescription,
    scale: scales.medium, // Use medium scale for export
    region: patchGeometry,
    maxPixels: 1e13,
    crs: commonProjection,
    folder: "MultiScale_Locust_Export",
  });

  print("✅ Created multi-scale export task:", exportDescription);
  print("Season:", season);
  print("Landcover class:", landcover.getInfo());
  print(
    "Non-critical missing variables:",
    missingDataVariables.length > 0 ? missingDataVariables : "None"
  );

  // Visualize multi-scale features if this is the first feature
  if (featureIndex === 0) {
    visualizeMultiScaleFeatures(multiScaleData, patchGeometry);
  }
}

// Habitat Suitability Index (HSI) Calculation Function
function calculateHabitatSuitabilityIndex(multiScaleImage) {
  // Weight different environmental variables based on locust ecology
  var hsiComponents = {
    vegetation: multiScaleImage.select("NDVI_.*").mean(),
    temperature: multiScaleImage.select("LST_Day_.*").mean(),
    moisture: multiScaleImage.select("sm_surface_.*").mean(),
    precipitation: multiScaleImage.select("precipitation_.*").mean(),
    elevation: multiScaleImage.select("elevation").multiply(0.1),
  };

  // Calculate composite HSI using weighted average
  var hsi = ee.Image.constant(0)
    .add(hsiComponents.vegetation.multiply(0.3)) // Vegetation most important
    .add(hsiComponents.temperature.multiply(0.2))
    .add(hsiComponents.moisture.multiply(0.2))
    .add(hsiComponents.precipitation.multiply(0.2))
    .add(hsiComponents.elevation.multiply(0.1))
    .rename("habitat_suitability_index");

  return hsi.clamp(0, 1); // Normalize between 0 and 1
}

// Data Completeness Validation Function
function hasCriticalDataMissing() {
  // Define critical variables that must be present
  var criticalVariables = [
    "NDVI_30",
    "LST_Day_1km_30",
    "sm_surface_30",
    "precipitation_30",
    "u_component_of_wind_10m_30",
    "elevation",
  ];

  // Check for missing critical variables
  var missingCritical = criticalVariables.some(function (variable) {
    return missingDataVariables.includes(variable);
  });

  // Allow some non-critical variables, but flag if too many are missing
  var totalMissingThreshold = 5;
  var totalMissing = missingDataVariables.length;

  return missingCritical || totalMissing > totalMissingThreshold;
}

// Visualization Function for Multi-Scale Features
function visualizeMultiScaleFeatures(multiScaleData, geometry) {
  // Select key visualization bands
  var visParams = {
    ndvi: {
      bands: ["NDVI_30"],
      min: 0,
      max: 1,
      palette: ["blue", "white", "green"],
    },
    temperature: {
      bands: ["LST_Day_1km_30"],
      min: 250,
      max: 350,
      palette: ["blue", "white", "red"],
    },
    hsi: {
      bands: ["habitat_suitability_index"],
      min: 0,
      max: 1,
      palette: ["red", "yellow", "green"],
    },
  };

  // Add layers to map for exploration
  Map.centerObject(geometry, 8);
  Map.addLayer(multiScaleData.select("NDVI_30"), visParams.ndvi, "NDVI");
  Map.addLayer(
    multiScaleData.select("LST_Day_1km_30"),
    visParams.temperature,
    "Land Surface Temperature"
  );

  // Optional: Add habitat suitability index if calculated
  try {
    var hsi = calculateHabitatSuitabilityIndex(multiScaleData);
    Map.addLayer(hsi, visParams.hsi, "Habitat Suitability Index");
  } catch (e) {
    print("Could not calculate HSI for visualization:", e);
  }
}

// Main Execution Function
function main() {
  // Load locust observation dataset
  var locustData = ee.FeatureCollection(
    "projects/desert-locust-forcast/assets/FAO_filtered_data_2000"
  );

  // Filter data for specific region and time period
  var filteredData = locustData
    .filter(ee.Filter.inList("Country", ["Ethiopia"]))
    .filter(ee.Filter.greaterThan("year", 2010))
    .filter(ee.Filter.lessThan("year", 2022));

  // Limit number of features for initial testing
  var sampleSize = 50;
  var sampledData = filteredData
    .randomColumn()
    .sort("random")
    .limit(sampleSize);

  // Process each feature
  sampledData.toList(sampleSize).evaluate(function (features) {
    features.forEach(function (feature, index) {
      createMultiScaleExportTask(index, feature);
    });
  });
}

// Optional: Logging and Error Tracking
function initializeLogging() {
  // Setup logging mechanism
  var logFile = ee.FeatureCollection([
    ee.Feature(null, {
      timestamp: Date.now(),
      message: "Locust Movement Prediction - Multi-Scale Analysis Started",
    }),
  ]);

  // Export log to Google Drive
  Export.table.toDrive({
    collection: logFile,
    description: "Locust_Prediction_Log_" + Date.now(),
    fileFormat: "CSV",
  });
}

// Configuration Object
var CONFIG = {
  MIN_OBSERVATIONS: 10,
  MAX_CLOUD_COVER: 20,
  BUFFER_RADIUS: 5000, // meters
  SCALES: {
    COARSE: 1000,
    MEDIUM: 250,
    FINE: 30,
  },
  DEBUG_MODE: true,
};

// Run the main script
try {
  // Initialize logging if in debug mode
  if (CONFIG.DEBUG_MODE) {
    initializeLogging();
  }

  // Execute main analysis
  main();
} catch (error) {
  print("Critical Error in Locust Movement Prediction:", error);
}
