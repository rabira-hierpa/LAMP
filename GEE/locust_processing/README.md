# Locust Processing Package

A modular package for processing and exporting locust data from Google Earth Engine.

## Package Structure

```
GEE/locust_processing/
├── __init__.py            # Package initialization
├── __main__.py            # Main entry point
├── config.py              # Configuration constants and settings
├── cli.py                 # Command-line interface
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── logging_utils.py   # Logging setup
│   ├── ee_utils.py        # Earth Engine initialization
│   └── geo_utils.py       # Geographic utilities
├── processing/            # Data processing functionality
│   ├── __init__.py
│   ├── indices.py         # Index calculations (VHI, TCI, TVDI)
│   ├── extraction.py      # Data extraction functions
│   └── export.py          # Export task creation
├── task_management/       # Task management
│   ├── __init__.py
│   └── task_manager.py    # TaskManager class
└── data/                  # Data handling
    ├── __init__.py
    └── progress.py        # Progress tracking
```

## Installation

The package is designed to be used as a local module. Clone the repository and ensure you have the required dependencies installed:

```bash
pip install earthengine-api numpy pandas
```

## Usage

### Running from the Command Line

You can run the package directly from the command line:

```bash
python -m GEE.locust_processing [options]
```

### Command-Line Options

- `--test`: Run with a single test point
- `--batch-size`: Number of features to process in one batch (default: 250)
- `--start-index`: Index to start processing from (default: 0)
- `--max-features`: Maximum number of features to process
- `--presence-only`: Process only presence points
- `--absence-only`: Process only absence points
- `--progress-file`: File to save/load progress (default: 'locust_export_progress.json')
- `--log-file`: Log file name (default: 'locust_export.log')

### Usage Examples

Run in test mode with a single point:

```bash
python -m GEE.locust_processing --test --start-index 42
```

Process 100 features starting at index 50:

```bash
python -m GEE.locust_processing --start-index 50 --max-features 100
```

Process only presence points:

```bash
python -m GEE.locust_processing --presence-only
```

### Importing in Python Code

You can also import and use the package in your Python code:

```python
from GEE.locust_processing.utils.ee_utils import initialize_ee
from GEE.locust_processing.processing.export import create_export_task

# Initialize Earth Engine
initialize_ee()

# ... your code here
```

## Configuration

Edit the `config.py` file to adjust settings like:

- Earth Engine dataset paths
- Export settings (scale, projection, etc.)
- Buffer sizes
- Task management parameters

## Extending the Package

The modular structure makes it easy to extend the package with new functionality:

1. Add new processing functions in the appropriate module
2. Update the command-line interface if needed
3. Update the configuration file with any new settings
