#!/bin/bash

REPO_OWNER="kakaobrain"
REPO_NAME="kor-nlu-datasets"
BRANCH="master"
BASE_PATH="KorSTS"
FILES=("sts-test.tsv" "sts-train.tsv" "sts-dev.tsv")
OUTPUT_DIR="."

for FILE in "${FILES[@]}"; do
    FILE_URL="https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/${BRANCH}/${BASE_PATH}/${FILE}"
    wget -O ${OUTPUT_DIR}/${FILE} ${FILE_URL}
    
    if [ $? -eq 0 ]; then
        echo "File downloaded successfully: ${OUTPUT_DIR}/${FILE}"
    else
        echo "Failed to download file: ${FILE_URL}"
    fi
done
