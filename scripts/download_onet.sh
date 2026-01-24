#!/bin/bash
# Download O*NET database files
# Usage: ./scripts/download_onet.sh

set -e

ONET_VERSION="29_0"
DOWNLOAD_URL="https://www.onetcenter.org/dl_files/database/db_${ONET_VERSION}_text.zip"
DATA_DIR="data/onet_raw"
ZIP_FILE="${DATA_DIR}/db_${ONET_VERSION}_text.zip"

echo "=== O*NET Database Download Script ==="
echo "Version: ${ONET_VERSION}"
echo "Target directory: ${DATA_DIR}"

# Create directory if needed
mkdir -p "${DATA_DIR}"

# Check if already downloaded
if [ -f "${DATA_DIR}/Occupation Data.txt" ]; then
    echo "O*NET database already exists in ${DATA_DIR}"
    echo "Delete the directory to re-download."
    exit 0
fi

# Download
echo "Downloading O*NET database..."
curl -L -o "${ZIP_FILE}" "${DOWNLOAD_URL}"

# Extract
echo "Extracting files..."
unzip -o "${ZIP_FILE}" -d "${DATA_DIR}"

# Clean up zip file
rm "${ZIP_FILE}"

# List key files
echo ""
echo "=== Downloaded files ==="
ls -la "${DATA_DIR}"/*.txt | head -20

echo ""
echo "=== Key files for persona generation ==="
echo "- Occupation Data.txt: Job titles and descriptions"
echo "- Interests.txt: RIASEC scores per occupation"
echo "- Tasks.txt: Job-specific tasks"
echo "- Skills.txt: Required skills"
echo ""
echo "Download complete!"
