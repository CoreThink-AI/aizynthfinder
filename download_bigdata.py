"""Download required bigdata files from GCS at container startup.

Files are cached in BIGDATA_DIR for the lifetime of the container instance.
With min-instances=1 the instance stays warm between requests, so this
download runs once per cold start.
"""
from __future__ import annotations

import os
from pathlib import Path

GCS_BUCKET = os.environ.get("GCS_BIGDATA_BUCKET", "biochem-db-by-hobs")
GCS_PREFIX = os.environ.get("GCS_BIGDATA_PREFIX", "aizynthfinder/bigdata")
BIGDATA_DIR = Path(os.environ.get("BIGDATA_DIR", "/app/bigdata"))

# Files required by config_production.yml
REQUIRED_FILES = [
    "uspto_model.onnx",
    "uspto_templates.csv.gz",
    "uspto_ringbreaker_model.onnx",
    "uspto_ringbreaker_templates.csv.gz",
    "uspto_filter_model.onnx",
    "zinc_stock.hdf5",
]


def main() -> None:
    BIGDATA_DIR.mkdir(parents=True, exist_ok=True)

    from google.cloud import storage  # imported here so tests can skip GCS

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    for filename in REQUIRED_FILES:
        local = BIGDATA_DIR / filename
        if local.exists():
            size_mb = local.stat().st_size / 1_000_000
            print(f"[bigdata] cached: {filename} ({size_mb:.0f} MB)", flush=True)
            continue
        blob = bucket.blob(f"{GCS_PREFIX}/{filename}")
        blob.reload()
        size_mb = (blob.size or 0) / 1_000_000
        print(f"[bigdata] downloading {filename} ({size_mb:.0f} MB) ...", flush=True)
        blob.download_to_filename(str(local))
        actual_mb = local.stat().st_size / 1_000_000
        print(f"[bigdata] done: {filename} ({actual_mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
