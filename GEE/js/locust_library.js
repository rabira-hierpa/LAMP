/**
 * Unified Desert Locust Analysis Library for Google Earth Engine
 * 
 * This library provides unified dataset preparation functions, multi-temporal
 * aggregation utilities, and common visualization templates for desert locust 
 * monitoring and prediction.
 */

// Module pattern to avoid polluting global namespace
var LocustLib = (function() {
  
  // Internal configuration - can be overridden with setConfig
  var config = {
    // Default region
    region: ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
      .filter(ee.Filter.eq("country_na", "Ethiopia")),
    
    // Common export parameters
    exportScale: 250,
    exportCrs: 'EPSG:4326',
    exportFolder: 'Locust_Exported_Images',
    maxPixels: 1e13,
    
    // Default dataset paths
    datasets: {
      faoReport: 'projects/desert-locust-forcast/assets/FAO_DL_data_extracted_2015',
      modisNdvi: 'MODIS/061/MOD13Q1',
      modisLst: 'MODIS/061/MOD11A2',
      precipitation: 'UCSB-CHG/CHIRPS/DAILY',
      wind: 'ECMWF/ERA5/DAILY',
      soilMoisture: 'NASA/SMAP/SPL4SMGP/007',
      landCover: 'MODIS/006/MCD12Q1',
      elevation: 'USGS/SRTMGL1_003'
    },
    
    // Multi-scale parameters
    scales: {
      coarse: 1000,  // 1km for climate patterns
      medium: 250,   // 250m for MODIS
      fine: 30       // 30m for Landsat/Sentinel
    },
    
    // Visualization parameters
    visParams: {
      ndvi: {min: 0, max: 1, palette: ['brown', 'yellow', 'green']},
      lst: {min: 290, max: 320, palette: ['blue', 'yellow', 'red']},
      precipitation: {min: 0, max: 100, palette: ['white', 'blue', 'purple']},
      vhi: {min: 0, max: 1, palette: ['red', 'yellow', 'green']}
    }
  };
  
  // ============= Dataset Preparation Functions =============
  
  /**
   * Get date ranges for time-lagged analysis
   * @param {ee.Date} date - Reference date
   * @param {Array<number>} lags - Array of lag days 
   * @return {Object} Dictionary of date ranges by lag period
   */
  function getDateRanges(date, lags) {
    var ranges = {'0': date};
    
    lags.forEach(function(lag) {
      ranges[lag.toString()] = date.advance(-lag, 'days');
    });
    
    return ranges;
  }
  
  /**
   * Extract NDVI data with error handling and fallbacks
   * @param {ee.Geometry} geometry - Area of interest
   * @param {Object} dateRange - Date range dictionary
   * @param {string} lag - Lag period key
   * @return {ee.Image} NDVI image
   */
  function extractNdvi(geometry, dateRange, lag) {
    var collection = ee.ImageCollection(config.datasets.modisNdvi)
      .filterBounds(geometry)
      .filterDate(dateRange[lag], dateRange['0']);
      
    // Check if collection is empty
    var collectionSize = collection.size();
    
    return ee.Algorithms.If(
      collectionSize.gt(0),
      collection.select(['NDVI']).mean().rename('NDVI_' + lag),
      ee.Image(0).rename('NDVI_' + lag) // Fallback value
    );
  }
  
  /**
   * Extract LST data with error handling and fallbacks
   * @param {ee.Geometry} geometry - Area of interest
   * @param {Object} dateRange - Date range dictionary
   * @param {string} lag - Lag period key
   * @return {ee.Image} LST image
   */
  function extractLst(geometry, dateRange, lag) {
    var collection = ee.ImageCollection(config.datasets.modisLst)
      .filterBounds(geometry)
      .filterDate(dateRange[lag], dateRange['0']);
      
    // Check if collection is empty
    var collectionSize = collection.size();
    
    return ee.Algorithms.If(
      collectionSize.gt(0),
      collection.select(['LST_Day_1km']).mean().rename('LST_Day_1km_' + lag),
      ee.Image(0).rename('LST_Day_1km_' + lag) // Fallback value
    );
  }
  /**
   * Extract precipitation data with error handling and fallbacks
   * @param {ee.Geometry} geometry - Area of interest
   * @param {Object} dateRange - Date range dictionary
   * @param {string} lag - Lag period key
   * @return {ee.Image} Precipitation image
   */
  function extractPrecipitation(geometry, dateRange, lag) {
    var collection = ee.ImageCollection(config.datasets.precipitation)
      .filterBounds(geometry)
      .filterDate(dateRange[lag], dateRange['0']);
      
    // Check if collection is empty
    var collectionSize = collection.size();
    
    return ee.Algorithms.If(
      collectionSize.gt(0),
      collection.select(['precipitation']).sum().rename('precipitation_' + lag),
      ee.Image(0).rename('precipitation_' + lag) // Fallback value
    );
  }
  
  /**
   * Extract wind components with fallbacks between ERA5 and NCEP
   * @param {ee.Geometry} geometry - Area of interest
   * @param {Object} dateRange - Date range dictionary
   * @param {string} lag - Lag period key
   * @return {Object} Dictionary with 'u' and 'v' wind components
   */
  function extractWindComponents(geometry, dateRange, lag) {
    // Try ERA5 first (preferred source)
    var era5Collection = ee.ImageCollection('ECMWF/ERA5/DAILY')
      .filterBounds(geometry)
      .filterDate(dateRange[lag], dateRange['0']);
      
    var era5Size = era5Collection.size();
    
    // If ERA5 is empty, try NCEP as fallback
    return ee.Algorithms.If(
      era5Size.gt(0),
      {
        'u': era5Collection.select(['u_component_of_wind_10m']).mean()
          .rename('u_component_of_wind_10m_' + lag),
        'v': era5Collection.select(['v_component_of_wind_10m']).mean()
          .rename('v_component_of_wind_10m_' + lag)
      },
      ee.Algorithms.If(
        ee.ImageCollection('NCEP_DOE_II/daily_averages')
          .filterBounds(geometry)
          .filterDate(dateRange[lag], dateRange['0'])
          .size().gt(0),
        {
          'u': ee.ImageCollection('NCEP_DOE_II/daily_averages')
            .filterBounds(geometry)
            .filterDate(dateRange[lag], dateRange['0'])
            .select(['uwnd_10m']).mean()
            .rename('u_component_of_wind_10m_' + lag),
          'v': ee.ImageCollection('NCEP_DOE_II/daily_averages')
            .filterBounds(geometry)
            .filterDate(dateRange[lag], dateRange['0'])
            .select(['vwnd_10m']).mean()
            .rename('v_component_of_wind_10m_' + lag)
        },
        // Final fallback if both sources fail
        {
          'u': ee.Image(0).rename('u_component_of_wind_10m_' + lag),
          'v': ee.Image(0).rename('v_component_of_wind_10m_' + lag)
        }
      )
    );
  }
  
  // ============= Index Calculation Functions =============
  
  /**
   * Calculate Temperature Condition Index (TCI)
   * @param {ee.Image} image - Image with LST bands
   * @return {ee.Image} Image with TCI bands added
   */
  function calculateTci(image) {
    var tci30 = image.select('LST_Day_1km_30').subtract(273.15).multiply(0.1)
      .rename('TCI_30');
    var tci60 = image.select('LST_Day_1km_60').subtract(273.15).multiply(0.1)
      .rename('TCI_60');
    var tci90 = image.select('LST_Day_1km_90').subtract(273.15).multiply(0.1)
      .rename('TCI_90');
      
    return image.addBands([tci30, tci60, tci90]);
  }
  
  /**
   * Calculate Vegetation Health Index (VHI)
   * @param {ee.Image} image - Image with NDVI and TCI bands
   * @return {ee.Image} Image with VHI bands added
   */
  function calculateVhi(image) {
    var vhi30 = image.select('NDVI_30').multiply(0.5)
      .add(image.select('TCI_30').multiply(0.5))
      .rename('VHI_30');
    var vhi60 = image.select('NDVI_60').multiply(0.5)
      .add(image.select('TCI_60').multiply(0.5))
      .rename('VHI_60');
    var vhi90 = image.select('NDVI_90').multiply(0.5)
      .add(image.select('TCI_90').multiply(0.5))
      .rename('VHI_90');
      
    return image.addBands([vhi30, vhi60, vhi90]);
  }
  
  /**
   * Calculate Temperature Vegetation Dryness Index (TVDI)
   * @param {ee.Image} image - Image with NDVI and LST bands
   * @return {ee.Image} Image with TVDI bands added
   */
  function calculateTvdi(image) {
    var tvdi30 = image.select('LST_Day_1km_30').subtract(273.15)
      .divide(image.select('NDVI_30').multiply(10).add(273.15))
      .rename('TVDI_30');
    var tvdi60 = image.select('LST_Day_1km_60').subtract(273.15)
      .divide(image.select('NDVI_60').multiply(10).add(273.15))
      .rename('TVDI_60');
    var tvdi90 = image.select('LST_Day_1km_90').subtract(273.15)
      .divide(image.select('NDVI_90').multiply(10).add(273.15))
      .rename('TVDI_90');
      
    return image.addBands([tvdi30, tvdi60, tvdi90]);
  }
  
  /**
   * Calculate all indices for an image with the required bands
   * @param {ee.Image} image - Image with required input bands
   * @return {ee.Image} Image with all indices added
   */
  function calculateAllIndices(image) {
    image = calculateTci(image);
    image = calculateVhi(image);
    image = calculateTvdi(image);
    return image;
  }
  
  // Public API
  return {
    // Config management
    getConfig: function() { return config; },
    setConfig: function(newConfig) { 
      // Merge new config with existing config
      for (var key in newConfig) {
        if (typeof newConfig[key] === 'object' && newConfig[key] !== null) {
          config[key] = Object.assign({}, config[key], newConfig[key]);
        } else {
          config[key] = newConfig[key]; 
        }
      }
      return config;
    },
    
    // Dataset preparation functions
    getDateRanges: getDateRanges,
    extractNdvi: extractNdvi,
    extractLst: extractLst,
    extractPrecipitation: extractPrecipitation,
    extractWindComponents: extractWindComponents,
    
    // Index calculation functions
    calculateTci: calculateTci,
    calculateVhi: calculateVhi, 
    calculateTvdi: calculateTvdi,
    calculateAllIndices: calculateAllIndices,
    
    // Export function
    exportImage: function(image, region, description, scale, crs, folder) {
      return Export.image.toDrive({
        image: image,
        description: description || 'locust_export',
        scale: scale || config.exportScale,
        region: region || config.region.geometry(),
        crs: crs || config.exportCrs,
        maxPixels: config.maxPixels,
        folder: folder || config.exportFolder
      });
    }
  };
})(); // End module pattern

// Make library available in global scope
exports = exports || {};
exports.LocustLib = LocustLib;
