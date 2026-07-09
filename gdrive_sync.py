"""
gdrive_sync.py
--------------
Syncs PDFs from a public Google Drive folder to the local knowledge_base_dir.
Uses gdown which works with publicly shared ("anyone with the link") folders
without requiring any OAuth credentials or a Google Cloud project.

The sync is incremental:
  - New files on Drive  -> downloaded and added to kb_dir
  - Changed files       -> re-downloaded (detected by file size difference)
  - Deleted from Drive  -> removed from kb_dir
  - Unchanged files     -> skipped entirely

A lightweight gdrive_manifest.json tracks {filename: size_bytes} for the
last successful sync, so we only touch files that actually changed.
"""

import json
import logging
import os
import re
import shutil
import tempfile

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GDRIVE_MANIFEST_FILE = os.path.join(BASE_DIR, "gdrive_manifest.json")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _extract_folder_id(drive_url: str) -> str:
    """Extract the raw folder ID from a Google Drive URL."""
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", drive_url)
    if match:
        return match.group(1)
    # Assume it is already a bare ID
    return drive_url.split("?")[0].strip("/").split("/")[-1]


def _load_gdrive_manifest() -> dict:
    if not os.path.exists(GDRIVE_MANIFEST_FILE):
        return {}
    try:
        with open(GDRIVE_MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_gdrive_manifest(manifest: dict) -> None:
    with open(GDRIVE_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Main sync entry point
# ---------------------------------------------------------------------------

def sync_drive_to_local(
    drive_url: str,
    kb_dir: str,
    status_callback=None,
) -> dict:
    """
    Sync all PDF files from a public Google Drive folder into kb_dir.

    Parameters
    ----------
    drive_url      : Full Google Drive folder URL or bare folder ID.
    kb_dir         : Local directory to sync PDFs into.
    status_callback: Optional callable(msg: str) used to report progress.

    Returns
    -------
    dict with keys:
        added     : list[str]  - filenames newly added or updated
        unchanged : list[str]  - filenames skipped (no change)
        deleted   : list[str]  - filenames removed (no longer on Drive)
        errors    : list[str]  - error messages if anything failed
    """
    import gdown  # imported here so the rest of the module loads without it

    folder_id = _extract_folder_id(drive_url)
    os.makedirs(kb_dir, exist_ok=True)

    result: dict = {"added": [], "unchanged": [], "deleted": [], "errors": []}
    old_manifest = _load_gdrive_manifest()

    print(f"\n[GDRIVE SYNC] Starting sync for folder ID: {folder_id} ...", flush=True)
    if status_callback:
        status_callback(f"Connecting to Google Drive folder …")

    # gdown needs a temporary staging directory so we can compare sizes before
    # overwriting the live kb_dir files that Chroma might be reading.
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if status_callback:
                status_callback("Downloading files from Google Drive …")

            print("[GDRIVE SYNC] Downloading folder contents using gdown...", flush=True)
            downloaded = gdown.download_folder(
                id=folder_id,
                output=tmp_dir,
                quiet=True,
                use_cookies=False,
            )

            if not downloaded:
                downloaded = []

            # Only care about PDFs
            pdf_files = [
                p for p in downloaded
                if p and p.lower().endswith(".pdf") and os.path.exists(p)
            ]

            print(f"[GDRIVE SYNC] Found {len(pdf_files)} PDF files in Google Drive folder.", flush=True)
            new_manifest: dict = {}

            for src_path in pdf_files:
                fname = os.path.basename(src_path)
                fsize = os.path.getsize(src_path)
                dst_path = os.path.join(kb_dir, fname)
                new_manifest[fname] = fsize

                if old_manifest.get(fname) != fsize:
                    # New or changed file: copy to kb_dir
                    print(f"[GDRIVE SYNC] [UPDATE] Copying {fname} ({fsize} bytes) to local KB...", flush=True)
                    shutil.copy2(src_path, dst_path)
                    result["added"].append(fname)
                else:
                    result["unchanged"].append(fname)

            # Handle deletions: files that were in the last manifest but are
            # no longer returned by Drive.
            for fname, _ in old_manifest.items():
                if fname not in new_manifest:
                    local_path = os.path.join(kb_dir, fname)
                    if os.path.exists(local_path):
                        print(f"[GDRIVE SYNC] [DELETE] Removing local file {fname} (deleted on Drive)...", flush=True)
                        os.remove(local_path)
                    result["deleted"].append(fname)

            _save_gdrive_manifest(new_manifest)
            
            print(f"[GDRIVE SYNC] Completed: {len(result['added'])} added/updated, "
                  f"{len(result['unchanged'])} unchanged, {len(result['deleted'])} deleted.\n", flush=True)

        except Exception as exc:
            msg = str(exc)
            result["errors"].append(msg)
            print(f"[GDRIVE SYNC] [ERROR] Sync failed: {msg}", flush=True)
            logger.error("Drive sync failed: %s", msg, exc_info=True)

    return result
