import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from osgeo import gdal
import json

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def get_google_drive_service():
    """Get Google Drive service with proper authentication."""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def download_file_from_drive(service, file_id, output_path):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%")

    with open(output_path, 'wb') as f:
        f.write(fh.getvalue())


def extract_geotiff_metadata(geotiff_path):
    """Extract metadata from a GeoTIFF file."""
    dataset = gdal.Open(geotiff_path)
    if dataset is None:
        raise Exception(f"Could not open {geotiff_path}")

    metadata = {
        'driver': dataset.GetDriver().ShortName,
        'size': {
            'x': dataset.RasterXSize,
            'y': dataset.RasterYSize,
            'bands': dataset.RasterCount
        },
        'projection': dataset.GetProjection(),
        'geotransform': dataset.GetGeoTransform(),
        'metadata': dataset.GetMetadata(),
        'band_info': []
    }

    # Get band information
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        band_info = {
            'band': i,
            'data_type': gdal.GetDataTypeName(band.DataType),
            'no_data_value': band.GetNoDataValue(),
            'statistics': band.GetStatistics(True, True),
            'metadata': band.GetMetadata()
        }
        metadata['band_info'].append(band_info)

    dataset = None  # Close the dataset
    return metadata


def main():
    # Initialize Google Drive service
    service = get_google_drive_service()

    # Example: Replace with your Google Drive file ID
    file_id = 'YOUR_FILE_ID'
    output_path = 'downloaded_geotiff.tif'

    try:
        # Download the file
        print("Downloading file from Google Drive...")
        download_file_from_drive(service, file_id, output_path)

        # Extract metadata
        print("\nExtracting metadata...")
        metadata = extract_geotiff_metadata(output_path)

        # Print metadata in a readable format
        print("\nGeoTIFF Metadata:")
        print(json.dumps(metadata, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up downloaded file
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    main()
