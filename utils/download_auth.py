#!/usr/bin/env python3
"""
Complete Google Drive Folder Downloader - Downloads ALL Files
Uses Google Drive API to list files (no 50 limit) + gdown to download

Just run: python download_complete.py
"""

from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from tqdm import tqdm
import gdown
import subprocess
import sys
import os
import time
import json
import hashlib
import pickle
from pathlib import Path

# Auto-install dependencies


def install_package(package):
    print(f"📦 Installing {package}...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "-q"])


packages = {
    'gdown': 'gdown',
    'google.oauth2': 'google-auth',
    'googleapiclient': 'google-api-python-client',
    'google_auth_oauthlib': 'google-auth-oauthlib',
    'tqdm': 'tqdm'
}

for module, package in packages.items():
    try:
        __import__(module)
    except ImportError:
        install_package(package)


# ============================================================================
# CONFIGURATION
# ============================================================================
FOLDER_URL = "https://drive.google.com/drive/u/0/folders/1V9PdYcI0quQo7ipmqFPtpppagen2CHHY"
DOWNLOAD_TO = "./locust_data"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
# ============================================================================

CHECKPOINT_FILE = ".download_checkpoint.json"
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.pickle"


class CompleteDownloader:
    def __init__(self, folder_url, output_dir):
        self.folder_url = folder_url
        self.folder_id = self.extract_folder_id(folder_url)
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / CHECKPOINT_FILE

        self.service = None
        self.total_files = 0
        self.downloaded_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.start_time = time.time()

        self.checkpoint = self.load_checkpoint()

    def extract_folder_id(self, url):
        """Extract folder ID from URL"""
        return url.split('folders/')[-1].split('?')[0]

    def load_checkpoint(self):
        """Load checkpoint"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'downloaded_files': {}, 'failed_files': []}

    def save_checkpoint(self):
        """Save checkpoint"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)

    def authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None

        # Try to load saved credentials
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Check if credentials file exists
                if not os.path.exists(CREDENTIALS_FILE):
                    print("\n" + "="*70)
                    print("⚠️  AUTHENTICATION REQUIRED")
                    print("="*70)
                    print("\nThis folder requires authentication to list all files.")
                    print("Don't worry - it's a one-time setup!\n")
                    print("📝 Setup Steps:")
                    print("1. Go to: https://console.cloud.google.com/")
                    print("2. Create project → Enable Google Drive API")
                    print("3. Create OAuth credentials (Desktop app)")
                    print("4. Download JSON → Save as 'credentials.json' here")
                    print("5. Run this script again\n")
                    print("="*70)

                    # Try fallback to gdown method
                    print(
                        "\n💡 Attempting download without API (limited to ~50 files)...")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save credentials
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('drive', 'v3', credentials=creds)
        return True

    def list_all_files(self, folder_id=None):
        """List ALL files in folder recursively - NO LIMIT!"""
        if folder_id is None:
            folder_id = self.folder_id

        files = []
        page_token = None

        print(f"🔍 Scanning folder for all files...")

        while True:
            try:
                response = self.service.files().list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, size)',
                    pageSize=1000,  # Max per request
                    pageToken=page_token
                ).execute()

                batch = response.get('files', [])
                files.extend(batch)

                print(f"   Found {len(files)} files so far...", end='\r')

                page_token = response.get('nextPageToken')
                if not page_token:
                    break

            except Exception as e:
                print(f"\n⚠️  Error listing files: {e}")
                break

        print(f"\n✅ Found {len(files)} total files in folder!\n")
        return files

    def download_file_with_gdown(self, file_id, file_name, destination):
        """Download a single file using gdown"""
        try:
            file_path = destination / file_name

            # Check if already downloaded
            file_key = hashlib.md5(str(file_path).encode()).hexdigest()
            if file_key in self.checkpoint['downloaded_files']:
                if file_path.exists():
                    self.skipped_count += 1
                    return True

            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download URL
            url = f"https://drive.google.com/uc?id={file_id}"

            # Download
            gdown.download(url, str(file_path), quiet=True)

            if file_path.exists() and file_path.stat().st_size > 0:
                # Mark as downloaded
                self.checkpoint['downloaded_files'][file_key] = {
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.downloaded_count += 1
                return True
            else:
                self.failed_count += 1
                return False

        except Exception as e:
            self.failed_count += 1
            if file_name not in self.checkpoint['failed_files']:
                self.checkpoint['failed_files'].append(file_name)
            return False

    def download(self):
        """Main download function"""
        print("\n" + "="*70)
        print("🚀 COMPLETE GOOGLE DRIVE FOLDER DOWNLOADER")
        print("="*70)
        print(f"\n📂 Source: {self.folder_url}")
        print(f"💾 Destination: {self.output_dir}\n")

        # Try to authenticate
        print("🔐 Authenticating with Google Drive API...")
        authenticated = self.authenticate()

        if not authenticated:
            print("\n⚠️  Could not authenticate with Google Drive API")
            print("💡 Falling back to gdown (limited to ~50 files)\n")
            self.fallback_download()
            return

        print("✅ Authentication successful!\n")

        # List ALL files
        try:
            all_files = self.list_all_files()
            self.total_files = len(all_files)

            if self.total_files == 0:
                print("⚠️  No files found in folder!")
                return

            print(f"📊 Total files to process: {self.total_files}")
            already_downloaded = len(self.checkpoint['downloaded_files'])
            if already_downloaded > 0:
                print(f"✅ Already downloaded: {already_downloaded}")
                print(
                    f"📥 Remaining: {self.total_files - already_downloaded}\n")

            # Download files with progress bar
            print("⏳ Starting download...\n")

            with tqdm(total=self.total_files, desc="Overall Progress", unit="file") as pbar:
                for idx, file_info in enumerate(all_files, 1):
                    file_name = file_info['name']
                    file_id = file_info['id']
                    mime_type = file_info.get('mimeType', '')

                    # Skip folders
                    if mime_type == 'application/vnd.google-apps.folder':
                        pbar.update(1)
                        continue

                    # Update progress bar description
                    pbar.set_description(
                        f"[{idx}/{self.total_files}] {file_name[:40]}")

                    # Download
                    success = self.download_file_with_gdown(
                        file_id, file_name, self.output_dir)

                    # Update progress
                    pbar.update(1)

                    # Save checkpoint every 10 files
                    if idx % 10 == 0:
                        self.save_checkpoint()

            # Final save
            self.save_checkpoint()
            self.print_summary()

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Progress saved.")
            self.save_checkpoint()
            self.print_summary()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.save_checkpoint()

    def fallback_download(self):
        """Fallback to gdown method (limited)"""
        try:
            print("📥 Using gdown (limited to ~50 files)...")
            gdown.download_folder(
                url=self.folder_url,
                output=str(self.output_dir),
                quiet=False,
                remaining_ok=True
            )
            print("\n⚠️  Note: gdown can only download ~50 files")
            print("   For complete download, set up authentication!\n")
        except Exception as e:
            print(f"❌ Download failed: {e}")

    def print_summary(self):
        """Print summary"""
        duration = time.time() - self.start_time

        print("\n" + "="*70)
        print("📊 DOWNLOAD SUMMARY")
        print("="*70)
        print(f"\n✅ Process complete!")
        print(f"\n📊 Statistics:")
        print(f"   • Total files in folder: {self.total_files}")
        print(f"   • Files downloaded: {self.downloaded_count}")
        print(f"   • Files skipped (existing): {self.skipped_count}")
        print(f"   • Files failed: {self.failed_count}")
        print(f"   • Time taken: {duration/60:.1f} minutes")

        # Calculate total size
        total_size = 0
        if self.output_dir.exists():
            for f in self.output_dir.rglob('*'):
                if f.is_file() and f.name not in [CHECKPOINT_FILE]:
                    total_size += f.stat().st_size

        print(f"   • Total size: {self.format_size(total_size)}")
        print(f"\n📁 Location: {self.output_dir.absolute()}")

        if self.failed_count > 0:
            print(f"\n⚠️  {self.failed_count} files failed to download")
            print(f"   Run script again to retry")

        print("\n" + "="*70)
        print("\n💡 Verify: find ./locust_data -type f -name '*.tif' | wc -l\n")

    @staticmethod
    def format_size(bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0


def main():
    downloader = CompleteDownloader(FOLDER_URL, DOWNLOAD_TO)
    downloader.download()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(130)

# ============================================================================
# CONFIGURATION - Change these if needed
# ============================================================================
FOLDER_URL = "https://drive.google.com/drive/u/0/folders/1V9PdYcI0quQo7ipmqFPtpppagen2CHHY"
DOWNLOAD_TO = "./locust_data"  # Where to save files
# ============================================================================

CHECKPOINT_FILE = ".download_checkpoint.json"
SKIPPED_FILES_LOG = ".skipped_files.log"


class UnlimitedDownloader:
    def __init__(self, folder_url, output_dir):
        self.folder_url = folder_url
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / CHECKPOINT_FILE
        self.skipped_log_path = self.output_dir / SKIPPED_FILES_LOG

        self.stats = {
            'downloaded': 0,
            'total_size': 0,
            'start_time': time.time()
        }

        self.checkpoint = self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint data from previous runs"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                    print(
                        f"📝 Loaded checkpoint: {len(data.get('downloaded_files', []))} files already downloaded")
                    return data
            except:
                return {'downloaded_files': {}, 'last_run': None}
        return {'downloaded_files': {}, 'last_run': None}

    def save_checkpoint(self):
        """Save checkpoint data"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')

        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)

    def log_skipped_file(self, filename, reason):
        """Log files that were skipped"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.skipped_log_path, 'a') as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {reason} | {filename}\n")

    def file_hash(self, filepath):
        """Generate hash of file path for tracking"""
        return hashlib.md5(str(filepath).encode()).hexdigest()

    def scan_existing_files(self):
        """Scan and register existing files in checkpoint"""
        if not self.output_dir.exists():
            return

        for file_path in self.output_dir.rglob('*'):
            if file_path.is_file() and file_path.name not in [CHECKPOINT_FILE, SKIPPED_FILES_LOG]:
                file_key = self.file_hash(file_path)
                if file_key not in self.checkpoint['downloaded_files']:
                    size = file_path.stat().st_size
                    self.checkpoint['downloaded_files'][file_key] = {
                        'path': str(file_path),
                        'size': size,
                        'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }

    def download(self):
        """Main download function"""
        print("\n" + "="*70)
        print("🚀 UNLIMITED GOOGLE DRIVE FOLDER DOWNLOADER")
        print("="*70)
        print(f"\n📂 Source: {self.folder_url}")
        print(f"💾 Destination: {self.output_dir}")
        print("✨ No 50 file limit - downloads ALL files!\n")

        # Scan existing files first
        print("🔍 Scanning for existing files...")
        self.scan_existing_files()

        existing_count = len(self.checkpoint['downloaded_files'])
        if existing_count > 0:
            print(
                f"✅ Found {existing_count} files already downloaded (will skip)")

        if self.checkpoint.get('last_run'):
            print(f"📅 Last run: {self.checkpoint['last_run']}")

        print("\n⏳ Starting download with unlimited file support...\n")

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                print("📥 Downloading ALL files from folder...")
                print("    • No 50 file limit!")
                print("    • Files already downloaded will be skipped")
                print("    • Inaccessible files will be logged")
                print("    • Progress shown below:\n")

                # Download using patched gdown
                gdown.download_folder(
                    url=self.folder_url,
                    output=str(self.output_dir),
                    quiet=False,
                    remaining_ok=True,
                    resume=True
                )

                print("\n🔄 Updating checkpoint...")
                self.scan_existing_files()
                self.save_checkpoint()

                # Calculate final statistics
                self.calculate_stats()
                self.print_summary()
                return

            except KeyboardInterrupt:
                print("\n\n⚠️  Download interrupted by user (Ctrl+C)")
                print("💾 Progress saved to checkpoint")
                print("💡 Run the script again to resume!\n")
                self.scan_existing_files()
                self.save_checkpoint()
                sys.exit(130)

            except Exception as e:
                error_msg = str(e)

                # Check if it's a permission/access error
                if any(x in error_msg.lower() for x in ["permission", "cannot retrieve", "forbidden", "failed to retrieve"]):
                    print(
                        f"\n⚠️  Access error (attempt {retry_count + 1}/{max_retries})")

                    # Save progress
                    self.scan_existing_files()
                    self.save_checkpoint()

                    # Extract file ID if possible
                    if "id=" in error_msg:
                        file_id = error_msg.split("id=")[1].split()[0]
                        self.log_skipped_file(
                            f"file_id_{file_id}", "Permission denied or rate limited")

                    retry_count += 1

                    if retry_count < max_retries:
                        wait_time = 10 * retry_count
                        print(
                            f"   Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        print(
                            f"   Retrying... (attempt {retry_count + 1}/{max_retries})\n")
                    else:
                        print("\n✅ Downloaded all accessible files")
                        print("⚠️  Some files remain inaccessible")
                        self.calculate_stats()
                        self.print_summary()
                        return
                else:
                    print(f"\n❌ Error: {error_msg}")
                    self.scan_existing_files()
                    self.save_checkpoint()
                    retry_count += 1

                    if retry_count < max_retries:
                        print(
                            f"   Retrying... (attempt {retry_count + 1}/{max_retries})\n")
                        time.sleep(5)
                    else:
                        print("💡 Run the script again to resume.\n")
                        sys.exit(1)

    def calculate_stats(self):
        """Calculate download statistics"""
        total_size = 0
        file_count = 0

        if self.output_dir.exists():
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file() and file_path.name not in [CHECKPOINT_FILE, SKIPPED_FILES_LOG]:
                    file_count += 1
                    total_size += file_path.stat().st_size

        self.stats['downloaded'] = file_count
        self.stats['total_size'] = total_size
        self.stats['duration'] = time.time() - self.stats['start_time']

    def format_size(self, bytes):
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"

    def print_summary(self):
        """Print download summary"""
        duration = self.stats['duration']

        print("\n" + "="*70)
        print("📊 DOWNLOAD SUMMARY")
        print("="*70)
        print(f"\n✅ Download complete!")
        print(f"\n📊 Statistics:")
        print(f"   • Total files downloaded: {self.stats['downloaded']}")
        print(f"   • Total size: {self.format_size(self.stats['total_size'])}")
        print(
            f"   • Time taken: {duration/60:.1f} minutes ({duration:.0f} seconds)")

        if duration > 0 and self.stats['total_size'] > 0:
            speed = self.stats['total_size'] / duration
            print(f"   • Average speed: {self.format_size(speed)}/s")

        print(f"\n📁 Location: {self.output_dir.absolute()}")

        # Show skipped files if any
        if self.skipped_log_path.exists():
            with open(self.skipped_log_path, 'r') as f:
                lines = f.readlines()
                skipped_count = len(lines)
            if skipped_count > 0:
                print(f"\n⚠️  Note: {skipped_count} files were inaccessible")
                print(f"   Details: {self.skipped_log_path.name}")
                print(f"\n💡 To download these:")
                print(f"   1. Wait 30 minutes (rate limit)")
                print(f"   2. Run script again")

        print("\n" + "="*70)
        print("\n💡 Verify your download:")
        print(f"   • Count: find {self.output_dir} -type f | wc -l")
        print(f"   • Size: du -sh {self.output_dir}")
        print("\n")


def main():
    """Main entry point"""
    downloader = UnlimitedDownloader(FOLDER_URL, DOWNLOAD_TO)
    downloader.download()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
