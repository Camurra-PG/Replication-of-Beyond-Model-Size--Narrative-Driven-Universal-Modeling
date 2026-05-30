#!/usr/bin/env bash
# Download and extract the Retailrocket e-commerce dataset from Kaggle.
# Dataset: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

set -euo pipefail

# --- Configuration ---
DEST_DIR="${1:-retailrocket_data}"
DATASET_SLUG="retailrocket/ecommerce-dataset"
TEMP_DIR="$(mktemp -d /tmp/retailrocket_download_XXXXXX)"

cleanup() {
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

echo "Destination directory: ${DEST_DIR}"

# --- Step 1: Check if data already exists ---
if [ -f "${DEST_DIR}/events.csv" ] && [ -f "${DEST_DIR}/category_tree.csv" ] && \
   { [ -f "${DEST_DIR}/item_properties_part1.csv" ] || [ -f "${DEST_DIR}/item_properties.csv" ]; }; then
    echo "✅ Retailrocket data already exists in '${DEST_DIR}'. Nothing to do."
    exit 0
fi

# --- Step 2: Ensure Kaggle CLI is available ---
if ! command -v kaggle >/dev/null 2>&1; then
    echo "Kaggle CLI not found. Installing the official 'kaggle' Python package..."
    python3 -m pip install --user --quiet kaggle
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v kaggle >/dev/null 2>&1; then
    echo "❌ Kaggle CLI could not be found after installation."
    echo "   Install it manually with: python3 -m pip install --user kaggle"
    exit 1
fi

# --- Step 3: Check Kaggle authentication ---
# Supported by the current CLI: OAuth login, API-token environment variable,
# token file, or legacy kaggle.json credentials.
if [ -z "${KAGGLE_API_TOKEN:-}" ] && \
   [ ! -f "${HOME}/.kaggle/access_token" ] && \
   [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
    echo "❌ Kaggle authentication is not configured."
    echo "   Run one of the following once, then execute this script again:"
    echo "   1) kaggle auth login"
    echo "   2) export KAGGLE_API_TOKEN='YOUR_TOKEN'"
    echo "   3) Place kaggle.json at ~/.kaggle/kaggle.json and run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# --- Step 4: Download ZIP archive ---
echo "Downloading Kaggle dataset '${DATASET_SLUG}'..."
kaggle datasets download -d "${DATASET_SLUG}" -p "${TEMP_DIR}" --force

ARCHIVE="$(find "${TEMP_DIR}" -maxdepth 1 -type f -name '*.zip' -print -quit)"
if [ -z "${ARCHIVE}" ]; then
    echo "❌ Download finished, but no ZIP archive was found in '${TEMP_DIR}'."
    exit 1
fi

# --- Step 5: Extract into destination directory ---
echo "Extracting archive into '${DEST_DIR}'..."
mkdir -p "${DEST_DIR}"
unzip -oq "${ARCHIVE}" -d "${DEST_DIR}"

# Some archive versions may include an enclosing folder. Move the relevant CSVs up.
EVENTS_FILE="$(find "${DEST_DIR}" -type f -name 'events.csv' -print -quit)"
if [ -n "${EVENTS_FILE}" ] && [ "$(dirname "${EVENTS_FILE}")" != "${DEST_DIR}" ]; then
    SOURCE_DIR="$(dirname "${EVENTS_FILE}")"
    find "${SOURCE_DIR}" -maxdepth 1 -type f -name '*.csv' -exec mv -f {} "${DEST_DIR}/" \;
fi

# --- Step 6: Validate expected Retailrocket files ---
missing=0
for file in events.csv category_tree.csv; do
    if [ ! -f "${DEST_DIR}/${file}" ]; then
        echo "❌ Missing expected file: ${DEST_DIR}/${file}"
        missing=1
    fi
done

if [ ! -f "${DEST_DIR}/item_properties_part1.csv" ] && [ ! -f "${DEST_DIR}/item_properties.csv" ]; then
    echo "❌ Missing item-properties data in '${DEST_DIR}'."
    missing=1
fi

if [ "${missing}" -ne 0 ]; then
    echo "Extracted files found:"
    find "${DEST_DIR}" -maxdepth 2 -type f -printf '  %P\n' | sort
    exit 1
fi

echo "✅ Retailrocket download and extraction complete."
echo "   Files are in '${DEST_DIR}'."
find "${DEST_DIR}" -maxdepth 1 -type f -name '*.csv' -printf '   - %f\n' | sort
