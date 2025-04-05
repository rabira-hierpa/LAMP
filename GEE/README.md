# Google Earth Engine (GEE) Components

This directory contains tools and code for working with Google Earth Engine, primarily focused on locust data processing and environmental data extraction.

## Directory Structure

```
GEE/
├── js/                           # JavaScript code for Google Earth Engine Code Editor
│   ├── Locust_dataset_preparation_CSV.js     # CSV dataset preparation script
│   ├── Locust_image_dataset_prep.js          # Image dataset preparation script
│   └── multi-scale-temporal-dataset-preparation.js  # Multi-scale temporal dataset prep
├── locust_processing/            # Python package for locust data processing
│   ├── README.md                 # Detailed documentation for the Python package
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration settings
│   └── (see package README for more details)
└── dataset-preparation-geotiff.js   # GeoTIFF dataset preparation script
```

## Components

### JavaScript Code (js/)

These scripts are designed to be used in the Google Earth Engine Code Editor:

1. **Locust Dataset Preparation (CSV)**: Script for extracting data for CSV export
2. **Locust Image Dataset Preparation**: Script for creating image exports
3. **Multi-scale Temporal Dataset Preparation**: Script for preparing multi-temporal scale datasets

### Python Package (locust_processing/)

A comprehensive Python package for processing and exporting locust data from Google Earth Engine.

Features include:

- Environmental data extraction
- Parallel task execution
- Progress tracking and resumable exports
- Balanced sampling between presence/absence points
- Country filtering
- Dry run mode for simulation

See the [locust_processing README](./locust_processing/README.md) for detailed documentation.

## Usage

### JavaScript Code

1. Copy the script content from the relevant .js file
2. Paste it into the Google Earth Engine Code Editor (https://code.earthengine.google.com/)
3. Modify parameters as needed
4. Run the script

### Python Package

See the [locust_processing README](./locust_processing/README.md) for detailed installation and usage instructions.

Quick start:

```bash
# Install dependencies
pip install earthengine-api numpy pandas

# Run with default settings
python -m GEE.locust_processing

# Run in test mode
python -m GEE.locust_processing --test

# Run with country filtering and balanced sampling
python -m GEE.locust_processing --country Ethiopia --balanced-sampling
```

## Requirements

- Google Earth Engine account
- Python 3.6+ (for Python package)
- earthengine-api Python package
- numpy, pandas
