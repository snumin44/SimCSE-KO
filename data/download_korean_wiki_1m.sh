#!/bin/bash

URL="https://github.com/snumin44/SimCSE-KO/releases/download/v1.0.0/korean_wiki_1m.txt"
FILENAME="korean_wiki_1m.txt"

wget -O ${FILENAME} ${URL}

if [ $? -eq 0 ]; then
    echo "File downloaded successfully: ${FILENAME}"
else
    echo "Failed to download file: ${FILENAME}"
fi