#!/usr/bin/env python3
"""
Google Drive Folder Downloader - NO 50 FILE LIMIT
Automatically patches gdown to download folders with unlimited files.

Just run: python download_unlimited.py
"""

import subprocess
import sys
import os
import time
import json
import hashlib
from pathlib import Path

# Auto-install dependencies


def install_package(package):
    print(f"📦 Installing {package}...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "-q"])


try:
    import gdown
except ImportError:
    install_package("gdown")
    import gdown

try:
    from tqdm import tqdm
except ImportError:
    install_package("tqdm")
    from tqdm import tqdm

# ============================================================================
# PATCH GDOWN TO REMOVE 50 FILE LIMIT
# ============================================================================


def patch_gdown_limit():
    """Remove the 50 file limit from gdown"""
    try:
        import gdown.download_folder
        import inspect

        # Find the gdown installation path
        gdown_path = Path(inspect.getfile(gdown.download_folder)).parent
        download_folder_file = gdown_path / "download_folder.py"

        if download_folder_file.exists():
            # Read the file
            with open(download_folder_file, 'r') as f:
                content = f.read()

            # Check if already patched
            if 'MAX_NUMBER_FILES = 50' in content:
                print("🔧 Patching gdown to remove 50 file limit...")
                # Replace the limit
                new_content = content.replace(
                    'MAX_NUMBER_FILES = 50',
                    'MAX_NUMBER_FILES = 999999'
                )

                # Write back
                with open(download_folder_file, 'w') as f:
                    f.write(new_content)

                print("✅ Patch applied! gdown can now download unlimited files\n")
                return True
            else:
                print("✅ gdown already patched or using newer version\n")
                return True
    except Exception as e:
        print(f"⚠️  Could not patch gdown: {e}")
        print("   Will try to download anyway with remaining_ok flag\n")
        return False


# Apply patch before importing
patch_gdown_limit()

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
