#!/usr/bin/env bash
# Download and extract the Retailrocket e-commerce dataset from Kaggle.
# Dataset: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

set -euo pipefail

# --- Configuration ---
DEST_DIR="${1:-retailrocket_data}"
DATASET_SLUG="retailrocket/ecommerce-dataset"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/retailrocket_download_XXXXXX")"

cleanup() {
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

echo "Destination directory: ${DEST_DIR}"

# --- Step 1: Check if data already exists ---
if [ -f "${DEST_DIR}/events.csv" ] && \
   [ -f "${DEST_DIR}/category_tree.csv" ] && \
   [ -f "${DEST_DIR}/item_properties_part1.csv" ] && \
   [ -f "${DEST_DIR}/item_properties_part2.csv" ]; then
    echo "✅ Retailrocket data already exists in '${DEST_DIR}'. Nothing to do."
    exit 0
fi

# --- Step 2: Ensure Kaggle CLI is available ---
if ! command -v kaggle >/dev/null 2>&1; then
    echo "Kaggle CLI not found. Installing the official 'kaggle' Python package..."

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo "❌ Python was not found. Please install Python first."
        exit 1
    fi

    "${PYTHON_CMD}" -m pip install --user --quiet kaggle
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v kaggle >/dev/null 2>&1; then
    echo "❌ Kaggle CLI could not be found after installation."
    echo "   Install it manually with:"
    echo "   python -m pip install kaggle"
    exit 1
fi

# --- Step 3: Download ZIP archive ---
# Kaggle itself checks whether authentication is configured.
echo "Downloading Kaggle dataset '${DATASET_SLUG}'..."
echo "If authentication is missing, configure Kaggle once and run this script again."

kaggle datasets download \
    -d "${DATASET_SLUG}" \
    -p "${TEMP_DIR}" \
    --force

ARCHIVE="$(find "${TEMP_DIR}" -maxdepth 1 -type f -name '*.zip' -print -quit)"

if [ -z "${ARCHIVE}" ]; then
    echo "❌ Download finished, but no ZIP archive was found in '${TEMP_DIR}'."
    exit 1
fi

# --- Step 4: Extract into destination directory ---
echo "Extracting archive into '${DEST_DIR}'..."
mkdir -p "${DEST_DIR}"
unzip -oq "${ARCHIVE}" -d "${DEST_DIR}"

# In case the ZIP contains an enclosing directory, move CSV files to DEST_DIR.
EVENTS_FILE="$(find "${DEST_DIR}" -type f -name 'events.csv' -print -quit)"

if [ -n "${EVENTS_FILE}" ] && [ "$(dirname "${EVENTS_FILE}")" != "${DEST_DIR}" ]; then
    SOURCE_DIR="$(dirname "${EVENTS_FILE}")"
    find "${SOURCE_DIR}" -maxdepth 1 -type f -name '*.csv' \
        -exec mv -f {} "${DEST_DIR}/" \;
fi

# --- Step 5: Validate expected Retailrocket files ---
missing=0

for file in \
    events.csv \
    category_tree.csv \
    item_properties_part1.csv \
    item_properties_part2.csv
do
    if [ ! -f "${DEST_DIR}/${file}" ]; then
        echo "❌ Missing expected file: ${DEST_DIR}/${file}"
        missing=1
    fi
done

if [ "${missing}" -ne 0 ]; then
    echo "Extracted files found:"
    find "${DEST_DIR}" -maxdepth 2 -type f -name '*.csv' -print | sort
    exit 1
fi

echo "✅ Retailrocket download and extraction complete."
echo "   Files are in '${DEST_DIR}':"
find "${DEST_DIR}" -maxdepth 1 -type f -name '*.csv' -printf '   - %f\n' | sort