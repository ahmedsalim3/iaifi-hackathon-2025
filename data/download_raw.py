# ======================================================================
# Download the source and target datasets
#
# How to run?
#
# 1. Download both datasets
# python data/download.py
#
# 2. Download only source dataset (IllustrisTNG)
# python data/download.py --source
#
# 3. Download only target dataset (GZ2)
# python data/download.py --target
# ======================================================================

import argparse
import shutil
import requests
import zipfile
import gzip
from pathlib import Path
import gdown

def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

def unzip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    zip_path.unlink()

def gunzip(gz_path, out_path):
    with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

def download_target():
    out = Path("data/target")
    urls = {
        "images_gz2.zip": "https://zenodo.org/records/3565489/files/images_gz2.zip?download=1",
        "gz2_filename_mapping.csv": "https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1",
        "README.txt": "https://zenodo.org/records/3565489/files/README.txt?download=1",
        "gz2_hart16.csv.gz": "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
    }

    download(urls["images_gz2.zip"], out / "images_gz2.zip")
    unzip(out / "images_gz2.zip", out / "images_gz2")

    for name in ["gz2_filename_mapping.csv", "README.txt"]:
        download(urls[name], out / name)

    download(urls["gz2_hart16.csv.gz"], out / "gz2_hart16.csv.gz")
    gunzip(out / "gz2_hart16.csv.gz", out / "gz2_hart16.csv")

def download_source():
    out = Path("data/source")
    gdown.download(id="1BcyjybNJelNu14gyqXN8qXgw_taU96pW", output=str(out / "illustrisTNG.zip"), quiet=False)
    unzip(out / "illustrisTNG.zip", out / "temp")
    shutil.move(str(out / "temp/illustrisTNG"), out)
    shutil.rmtree(out / "temp")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download datasets")
    p.add_argument("--target", action="store_true", help="Download target data")
    p.add_argument("--source", action="store_true", help="Download source data")
    a = p.parse_args()

    if a.target or not (a.target or a.source):
        download_target()
    if a.source or not (a.target or a.source):
        download_source()
