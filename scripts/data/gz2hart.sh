#!/usr/bin/env bash
# This script will download the Galaxy Zoo 2 (GZ2) dataset
# based on Hart et al. (2016) morphological classifications:
#
# References:
# - paper: https://academic.oup.com/mnras/article/461/4/3663/2608720?login=false
# - data access: https://data.galaxyzoo.org/#section-8

set -e
readonly OUTPUT_DIR="data/target"
readonly GZ_HART_URL="https://gz2hart.s3.amazonaws.com"

mkdir -p "$OUTPUT_DIR"
pushd "$OUTPUT_DIR" > /dev/null

echo "Downloading and unzipping image data..."
wget --quiet --show-progress https://zenodo.org/records/3565489/files/images_gz2.zip?download=1 -O images_gz2.zip
unzip -q images_gz2.zip -d images_gz2
rm images_gz2.zip

echo "Downloading metadata files..."
wget --quiet --show-progress https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1 -O gz2_filename_mapping.csv
wget --quiet --show-progress https://zenodo.org/records/3565489/files/README.txt?download=1 -O README.txt

echo "Downloading and decompressing GZ2 HART data..."
wget --quiet --show-progress https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz -O gz2_hart16.csv.gz
gunzip gz2_hart16.csv.gz

popd > /dev/null
echo "files successfully downloaded and processed in '$OUTPUT_DIR'."
