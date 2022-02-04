#!/bin/bash

FILEID=$1
FILENAME=$2

curl -sc ./cookie.txt "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null

curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=`awk '/_warning_/ {print $NF}' ./cookie.txt`&id=${FILEID}" -o ${FILENAME}
