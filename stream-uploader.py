#!/usr/bin/env python3
"""
stream_upload_v2.py
-------------------
Stream files from a Google Drive folder to Harvard Dataverse in batches,
using the OFFICIAL Google Drive API with OAuth2 (instead of gdown).

Why v2:
  v1 used gdown, which makes anonymous requests to Drive's public-file endpoint.
  Google rate-limits that endpoint aggressively ("Cannot retrieve the public
  link... have had many accesses"). v1 also silently truncated folder listings
  above ~5500 files.

  v2 uses the authenticated Drive API, which has a ~1000 requests/100s quota
  per user (plenty for 18k files) and paginates correctly. You auth once via
  browser; the refresh token is cached and reused forever after.

Key properties (unchanged from v1):
  * Resume-safe:  JSON state file with atomic writes.
  * Disk-lean:    Downloads batch -> uploads each -> deletes locally.
  * Two-phase:    Phase 1 attempts every file. Phase 2 retries failures.
  * State-compatible: Reads your existing upload_state.json from v1.
                      36 already-uploaded files will NOT be re-uploaded.

New in v2:
  * Official Drive API client with OAuth2
  * Exponential backoff on 403/429 rate-limit responses
  * Correct pagination (will find all 18,871 files, not just 5,500)
  * --rescan: re-enumerate GDrive to pick up files v1 missed
  * --reset-failed: move status=failed files back to pending for retry

First-run setup (one-time, 5 minutes):
  1. https://console.cloud.google.com/ -> new project (or existing)
  2. APIs & Services -> Library -> search "Google Drive API" -> Enable
  3. APIs & Services -> OAuth consent screen -> External -> fill minimal fields
       -> add your email under "Test users"
  4. APIs & Services -> Credentials -> Create Credentials -> OAuth client ID
       -> Application type: Desktop app -> name it -> Create
  5. Download JSON -> save as credentials.json next to this script
  6. First run opens a browser asking you to grant read-only Drive access.
     Token is saved to token.json; subsequent runs are fully automatic.

Install:
  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2 \
              requests tqdm

Usage:
  export DATAVERSE_API_TOKEN="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  python stream_upload_v2.py \
      --persistent-id doi:10.7910/DVN/XXXXXX \
      --rescan --reset-failed      # first run after migrating from v1
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from tqdm import tqdm


# ----------------------------- Configuration ------------------------------

DEFAULTS = {
    "dataverse_url":        "https://dataverse.harvard.edu",
    "gdrive_folder_id":     "1V9PdYcI0quQo7ipmqFPtpppagen2CHHY",
    "batch_size":           100,
    "temp_dir":             "./temp_downloads",
    "state_file":           "./upload_state.json",
    "log_file":             "./upload.log",
    "credentials_file":     "./credentials.json",
    "token_file":           "./token.json",
    "max_retries":          3,
    "retry_backoff":        5,
    "request_timeout":      300,
    "sleep_between_files":  0.1,
    "download_chunk_mb":    10,
    "max_backoff":          64,     # cap exponential backoff at 64s
}

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# --------------------------- Filename parsing -----------------------------

FNAME_RE = re.compile(
    r"locust_(?P<date>\d{4}-\d{2}-\d{2})_label_(?P<label>[01])"
    r"_(?P<country>[^_]+)_idx_(?P<idx>\d+)\.tif",
    re.IGNORECASE,
)


def parse_filename(name: str) -> Optional[dict]:
    m = FNAME_RE.match(name)
    if not m:
        return None
    year = m.group("date")[:4]
    label = "presence" if m.group("label") == "1" else "absence"
    return {
        "year":            year,
        "date":            m.group("date"),
        "label":           label,
        "country":         m.group("country"),
        "idx":             m.group("idx"),
        "directory_label": f"{year}/{label}",
    }


# ----------------------------- State tracking ------------------------------

@dataclass
class FileState:
    file_id:    str
    name:       str
    status:     str = "pending"          # pending | uploaded | failed
    attempts:   int = 0
    error:      str = ""
    dv_file_id: Optional[int] = None


class StateManager:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.files: Dict[str, FileState] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        with open(self.state_file) as f:
            data = json.load(f)
        self.files = {
            fid: FileState(**fd) for fid, fd in data.get("files", {}).items()
        }
        logging.info("Resumed state: %d files tracked", len(self.files))

    def save(self) -> None:
        tmp = self.state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "files":        {fid: asdict(fs) for fid, fs in self.files.items()},
                    "last_updated": time.time(),
                },
                f, indent=2,
            )
        tmp.replace(self.state_file)

    def register(self, file_id: str, name: str) -> bool:
        """Add a file if not already tracked. Returns True if newly added."""
        if file_id not in self.files:
            self.files[file_id] = FileState(file_id=file_id, name=name)
            return True
        return False

    def mark(self, file_id: str, status: str, error: str = "",
             dv_file_id: Optional[int] = None) -> None:
        fs = self.files.get(file_id)
        if fs is None:
            return
        fs.status = status
        fs.attempts += 1
        if status == "uploaded":
            fs.dv_file_id = dv_file_id
            fs.error = ""
        else:
            fs.error = error

    def reset_failed(self) -> int:
        """Move all status=failed files back to pending. Returns count reset."""
        n = 0
        for fs in self.files.values():
            if fs.status == "failed":
                fs.status = "pending"
                fs.error = ""
                n += 1
        return n

    def pending(self) -> List[FileState]:
        return [fs for fs in self.files.values() if fs.status == "pending"]

    def retryable(self, max_attempts: int) -> List[FileState]:
        return [
            fs for fs in self.files.values()
            if fs.status == "failed" and fs.attempts < max_attempts
        ]

    def summary(self) -> Dict[str, int]:
        counts = {"pending": 0, "uploaded": 0, "failed": 0}
        for fs in self.files.values():
            counts[fs.status] = counts.get(fs.status, 0) + 1
        return counts


# --------------------------- Google Drive API ------------------------------

def get_drive_service(credentials_file: Path, token_file: Path):
    """OAuth2 flow. Caches and refreshes token automatically."""
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logging.info("Refreshing expired OAuth token...")
            creds.refresh(Request())
        else:
            if not credentials_file.exists():
                raise FileNotFoundError(
                    f"{credentials_file} not found. See the docstring of this "
                    "script for one-time setup instructions."
                )
            logging.info("Starting OAuth flow (browser will open)...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_file, "w") as t:
            t.write(creds.to_json())
        logging.info("Saved OAuth token to %s", token_file)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _execute_with_backoff(request, max_backoff: int):
    """Execute a Drive API request with exponential backoff on rate limits."""
    for attempt in range(8):                         # up to ~2 minutes total
        try:
            return request.execute()
        except HttpError as e:
            status = e.resp.status
            reason = ""
            try:
                err = json.loads(e.content).get("error", {})
                reason = err.get("errors", [{}])[0].get("reason", "")
            except Exception:                        # noqa: BLE001
                pass
            transient = status in (403, 429, 500, 502, 503, 504) and (
                status != 403 or reason in {
                    "userRateLimitExceeded", "rateLimitExceeded",
                    "quotaExceeded", "backendError",
                }
            )
            if not transient or attempt == 7:
                raise
            wait = min(2 ** attempt + random.random(), max_backoff)
            logging.warning(
                "Drive API %d (%s); backing off %.1fs (attempt %d/8)",
                status, reason or "transient", wait, attempt + 1,
            )
            time.sleep(wait)


def list_drive_folder(
    service,
    folder_id: str,
    max_backoff: int,
) -> List[dict]:
    """Paginated listing of a Drive folder. Returns all items, any extension."""
    all_items: List[dict] = []
    page_token: Optional[str] = None
    q = f"'{folder_id}' in parents and trashed = false"
    logging.info("Listing Google Drive folder %s...", folder_id)

    page = 0
    while True:
        req = service.files().list(
            q          = q,
            pageSize   = 1000,            # API max
            fields     = "nextPageToken, files(id, name, size, mimeType)",
            pageToken  = page_token,
            orderBy    = "name",          # deterministic order
            supportsAllDrives        = True,
            includeItemsFromAllDrives = True,
        )
        resp = _execute_with_backoff(req, max_backoff)
        batch = resp.get("files", [])
        all_items.extend(batch)
        page += 1
        logging.info("  page %d: +%d (total %d)", page, len(batch), len(all_items))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    logging.info("Total items listed: %d", len(all_items))
    return all_items


def download_drive_file(
    service,
    file_id: str,
    dest: Path,
    chunk_mb: int,
    max_backoff: int,
) -> bool:
    """Download one file in chunks with resilient retries. Returns True on success."""
    try:
        req = service.files().get_media(fileId=file_id)
        # Use a tmp path so a partial download on crash doesn't look complete
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        with open(tmp_dest, "wb") as fh:
            downloader = MediaIoBaseDownload(
                fh, req, chunksize=chunk_mb * 1024 * 1024
            )
            done = False
            while not done:
                for attempt in range(8):
                    try:
                        _, done = downloader.next_chunk()
                        break
                    except HttpError as e:
                        if e.resp.status in (403, 429, 500, 502, 503, 504) and attempt < 7:
                            wait = min(2 ** attempt + random.random(), max_backoff)
                            logging.warning(
                                "Download %s: HTTP %d, retrying in %.1fs",
                                file_id, e.resp.status, wait,
                            )
                            time.sleep(wait)
                        else:
                            raise
        tmp_dest.replace(dest)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:                           # noqa: BLE001
        logging.error("Download failed for %s: %s", file_id, e)
        # Clean up partial download if any
        for p in (dest, dest.with_suffix(dest.suffix + ".part")):
            if p.exists():
                try:    p.unlink()
                except: pass
        return False


# --------------------------- Dataverse upload ------------------------------

def upload_to_dataverse(
    filepath:        Path,
    dataverse_url:   str,
    persistent_id:   str,
    api_token:       str,
    directory_label: str,
    description:     str,
    timeout:         int,
) -> Optional[int]:
    """POST /api/datasets/:persistentId/add. Returns Dataverse file id or None."""
    endpoint  = f"{dataverse_url}/api/datasets/:persistentId/add"
    json_data = {
        "description":    description,
        "directoryLabel": directory_label,
        "categories":     ["Data"],
        "restrict":       "false",
    }
    try:
        with open(filepath, "rb") as fp:
            resp = requests.post(
                endpoint,
                params  = {"persistentId": persistent_id},
                headers = {"X-Dataverse-key": api_token},
                files   = {
                    "file":     (filepath.name, fp, "image/tiff"),
                    "jsonData": (None, json.dumps(json_data), "application/json"),
                },
                timeout = timeout,
            )
        if resp.status_code in (200, 201):
            try:
                return resp.json()["data"]["files"][0]["dataFile"]["id"]
            except (KeyError, IndexError, ValueError):
                return -1
        logging.error(
            "Upload failed %s: HTTP %d -- %s",
            filepath.name, resp.status_code, resp.text[:300].replace("\n", " "),
        )
        return None
    except requests.Timeout:
        logging.error("Upload timeout for %s", filepath.name)
        return None
    except Exception as e:                           # noqa: BLE001
        logging.error("Upload exception for %s: %s", filepath.name, e)
        return None


# ------------------------------ Pipeline -----------------------------------

class StreamingUploader:
    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.temp_dir = Path(cfg["temp_dir"])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.state    = StateManager(Path(cfg["state_file"]))
        self.service  = get_drive_service(
            Path(cfg["credentials_file"]),
            Path(cfg["token_file"]),
        )
        self._stop = False
        signal.signal(signal.SIGINT,  self._sig)
        signal.signal(signal.SIGTERM, self._sig)

    def _sig(self, signum, _frame):
        logging.warning("Signal %d received -- will stop after current file", signum)
        self._stop = True

    # .....................................................................

    def enumerate_and_merge(self, force: bool = False) -> None:
        """Enumerate GDrive and add any missing files. Idempotent."""
        if not force and len(self.state.files) > 0:
            logging.info("State already has %d files; use --rescan to force re-enumerate",
                         len(self.state.files))
            return
        items = list_drive_folder(
            self.service,
            self.cfg["gdrive_folder_id"],
            self.cfg["max_backoff"],
        )
        added = existing = skipped = 0
        for it in items:
            if not it["name"].lower().endswith(".tif"):
                skipped += 1
                continue
            if self.state.register(it["id"], it["name"]):
                added += 1
            else:
                existing += 1
        self.state.save()
        logging.info(
            "Enumeration merged: %d new, %d already tracked, %d non-tif skipped",
            added, existing, skipped,
        )

    # .....................................................................

    def _process_one(self, fs: FileState) -> bool:
        local = self.temp_dir / fs.name

        # 1. Download (if not already on disk)
        if not (local.exists() and local.stat().st_size > 0):
            ok = download_drive_file(
                self.service, fs.file_id, local,
                self.cfg["download_chunk_mb"], self.cfg["max_backoff"],
            )
            if not ok:
                self.state.mark(fs.file_id, "failed", error="download_failed")
                return False

        # 2. Parse filename for Dataverse folder layout
        meta = parse_filename(fs.name)
        if meta is None:
            dir_label   = "unparsed"
            description = f"GeoTIFF (filename did not match expected pattern): {fs.name}"
            logging.warning("Unparsable filename: %s", fs.name)
        else:
            dir_label   = meta["directory_label"]
            description = (
                f"Desert locust {meta['label']} observation recorded on "
                f"{meta['date']} in {meta['country']} (observation index "
                f"{meta['idx']}). 41x41-pixel window at 250m resolution with "
                f"59 environmental bands (vegetation, climate, soil, terrain)."
            )

        # 3. Upload
        dv_id = upload_to_dataverse(
            filepath       = local,
            dataverse_url  = self.cfg["dataverse_url"],
            persistent_id  = self.cfg["persistent_id"],
            api_token      = self.cfg["api_token"],
            directory_label = dir_label,
            description    = description,
            timeout        = self.cfg["request_timeout"],
        )
        if dv_id is None:
            self.state.mark(fs.file_id, "failed", error="upload_failed")
            return False

        # 4. Success: reclaim disk
        self.state.mark(fs.file_id, "uploaded", dv_file_id=dv_id)
        try:
            local.unlink()
        except OSError as e:
            logging.warning("Could not delete %s: %s", local, e)
        return True

    # .....................................................................

    def _process_batch(self, batch: List[FileState], desc: str) -> Dict[str, int]:
        stats = {"ok": 0, "fail": 0}
        for i, fs in enumerate(tqdm(batch, desc=desc, leave=False)):
            if self._stop:
                break
            ok = self._process_one(fs)
            stats["ok" if ok else "fail"] += 1
            if (i + 1) % 10 == 0:
                self.state.save()
            time.sleep(self.cfg["sleep_between_files"])
        self.state.save()
        return stats

    def _run_phase(self, files: List[FileState], phase: str) -> None:
        bs, total = self.cfg["batch_size"], len(files)
        if total == 0:
            logging.info("%s: nothing to do", phase)
            return
        logging.info("%s: %d files, batch size %d", phase, total, bs)
        for start in range(0, total, bs):
            if self._stop:
                logging.warning("%s: stop requested", phase)
                break
            batch   = files[start:start + bs]
            n_batch = (total + bs - 1) // bs
            label   = f"{phase} batch {start // bs + 1}/{n_batch}"
            stats   = self._process_batch(batch, label)
            logging.info(
                "%s done: %d ok, %d fail | overall %s",
                label, stats["ok"], stats["fail"], self.state.summary(),
            )

    # .....................................................................

    def run(self, rescan: bool = False, reset_failed: bool = False) -> None:
        self.enumerate_and_merge(force=rescan)

        if reset_failed:
            n = self.state.reset_failed()
            self.state.save()
            logging.info("Reset %d failed files to pending", n)

        self._run_phase(self.state.pending(), "PHASE 1")

        attempt = 1
        while not self._stop:
            retryable = self.state.retryable(self.cfg["max_retries"])
            if not retryable:
                break
            wait = self.cfg["retry_backoff"] * attempt
            logging.info(
                "PHASE 2.%d: %d failed files; waiting %ds before retry",
                attempt, len(retryable), wait,
            )
            time.sleep(wait)
            for fs in retryable:
                fs.status = "pending"
            self.state.save()
            self._run_phase(retryable, f"PHASE 2.{attempt}")
            attempt += 1

        logging.info("FINAL: %s", self.state.summary())
        perma = [
            fs.name for fs in self.state.files.values()
            if fs.status == "failed" and fs.attempts >= self.cfg["max_retries"]
        ]
        if perma:
            logging.warning("Permanently failed (%d): see failed_files.txt", len(perma))
            Path("failed_files.txt").write_text("\n".join(perma))


# --------------------------------- CLI -------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--persistent-id",      required=True,
                   help="Dataset DOI, e.g. doi:10.7910/DVN/XXXXXX")
    p.add_argument("--api-token",          default=os.environ.get("DATAVERSE_API_TOKEN", ""))
    p.add_argument("--gdrive-folder-id",   default=DEFAULTS["gdrive_folder_id"])
    p.add_argument("--dataverse-url",      default=DEFAULTS["dataverse_url"])
    p.add_argument("--batch-size",         type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--temp-dir",           default=DEFAULTS["temp_dir"])
    p.add_argument("--state-file",         default=DEFAULTS["state_file"])
    p.add_argument("--log-file",           default=DEFAULTS["log_file"])
    p.add_argument("--credentials-file",   default=DEFAULTS["credentials_file"],
                   help="OAuth client_secrets JSON from Google Cloud Console")
    p.add_argument("--token-file",         default=DEFAULTS["token_file"],
                   help="Cached OAuth token (created on first run)")
    p.add_argument("--max-retries",        type=int, default=DEFAULTS["max_retries"])
    p.add_argument("--request-timeout",    type=int, default=DEFAULTS["request_timeout"])
    p.add_argument("--retry-backoff",      type=int, default=DEFAULTS["retry_backoff"])
    p.add_argument("--max-backoff",        type=int, default=DEFAULTS["max_backoff"])
    p.add_argument("--download-chunk-mb",  type=int, default=DEFAULTS["download_chunk_mb"])
    p.add_argument("--sleep-between-files", type=float, default=DEFAULTS["sleep_between_files"])
    p.add_argument("--rescan",             action="store_true",
                   help="Force re-enumeration of the GDrive folder "
                        "(adds any missing files to state)")
    p.add_argument("--reset-failed",       action="store_true",
                   help="Before Phase 1, move all status=failed back to pending "
                        "(use this when migrating from v1)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_token:
        print("ERROR: supply --api-token or set DATAVERSE_API_TOKEN", file=sys.stderr)
        return 1

    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
        handlers = [
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Quiet the very noisy google libraries
    for noisy in ("googleapiclient", "google_auth_httplib2", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = {**DEFAULTS, **vars(args)}
    StreamingUploader(cfg).run(rescan=args.rescan, reset_failed=args.reset_failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())