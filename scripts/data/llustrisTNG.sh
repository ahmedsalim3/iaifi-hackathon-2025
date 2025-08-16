#!/usr/bin/env bash
# This script downloads the llustrisTNG dataset from a google drive link
# and unzips it in the data/source directory
# Make sure to install gdown first: pip install gdown
set -e

readonly OUTPUT_DIR="data/source"

mkdir -p "$OUTPUT_DIR"
pushd "$OUTPUT_DIR" > /dev/null

echo "Downloading and unzipping image data..."
gdown --id 1BcyjybNJelNu14gyqXN8qXgw_taU96pW -O "llustrisTNG.zip"
unzip -q "llustrisTNG.zip" -d "temp"
mv "temp/llustrisTNG"/* "."

rm -rf "temp"
rm -f "llustrisTNG.zip"

popd > /dev/null
echo "files successfully downloaded and processed in '$OUTPUT_DIR'."