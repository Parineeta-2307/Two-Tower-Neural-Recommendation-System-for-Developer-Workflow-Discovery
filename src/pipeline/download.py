"""
src/pipeline/download.py
────────────────────────
Downloads GitHub Archive event data (.json.gz files) for a
specified date/hour range and saves them to data/raw/.

GitHub Archive URL format:
  https://data.gharchive.org/YYYY-MM-DD-{H}.json.gz
  (H is 0–23, no zero-padding)

We download ~5 hours of data → roughly 8,000–15,000 active users
after filtering. That's enough to train our model.

Run with:
    python src/pipeline/download.py
Output:
    data/raw/*.json.gz  (compressed event files)
"""

import os
import urllib.request    # built-in: downloads files from URLs
import time              # built-in: for polite delays between requests

# ── Configuration ──────────────────────────────────────────────────────────────
# We download 5 hours from one day.
# Why Jan 15 2024? Weekday, high activity, representative data.
# Why 5 hours? Enough users without taking too long to download.

DATE   = "2024-01-15"
HOURS  = [9, 10, 11, 12, 13]   # 9am–1pm UTC (peak GitHub activity)
OUTPUT_DIR = "data/raw"

BASE_URL = "https://data.gharchive.org"


def download_hour(date, hour, output_dir):
    """
    Download one hour of GitHub Archive data.

    Parameters:
        date       : "YYYY-MM-DD" string
        hour       : integer 0–23
        output_dir : where to save the .json.gz file

    Returns:
        local file path if successful, None if failed
    """
    filename  = f"{date}-{hour}.json.gz"          # e.g. "2024-01-15-9.json.gz"
    url       = f"{BASE_URL}/{filename}"
    dest_path = os.path.join(output_dir, filename)

    # Skip if already downloaded (so re-running is safe)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  SKIP  {filename}  (already exists, {size_mb:.1f} MB)")
        return dest_path

    print(f"  GET   {url}")

    try:
        # urllib.request.urlretrieve downloads a URL to a local file
        # reporthook lets us show download progress
        def show_progress(block_count, block_size, total_size):
            if total_size > 0:
                downloaded = block_count * block_size
                pct = min(downloaded / total_size * 100, 100)
                # \r moves cursor to start of line (overwrites previous output)
                print(f"\r        {pct:5.1f}%  ({downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB)", end="")

        urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
        print()  # newline after progress line

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  DONE  {filename}  ({size_mb:.1f} MB)")
        return dest_path

    except Exception as e:
        print(f"\n  FAIL  {filename}: {e}")
        # Clean up partial file if download failed
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading GitHub Archive data")
    print(f"Date  : {DATE}")
    print(f"Hours : {HOURS}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"─" * 50)
    print(f"Expected size: ~100–200 MB total (compressed)")
    print(f"This will take 2–5 minutes depending on your connection.")
    print(f"─" * 50)
    print()

    downloaded = []
    failed     = []

    for hour in HOURS:
        result = download_hour(DATE, hour, OUTPUT_DIR)
        if result:
            downloaded.append(result)
        else:
            failed.append(hour)

        # Polite delay between requests — don't hammer the server
        if hour != HOURS[-1]:
            time.sleep(1)

    print()
    print(f"─" * 50)
    print(f"Downloaded : {len(downloaded)} files")
    if failed:
        print(f"Failed     : hours {failed} — try re-running, usually a network blip")
    print()
    print("Next step: python src/pipeline/extract.py")


if __name__ == "__main__":
    main()