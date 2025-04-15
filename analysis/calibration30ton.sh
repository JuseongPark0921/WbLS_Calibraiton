#!/bin/bash

# 사용법: . calibration.sh sumSPE30t.py YYMMDD AABB

if [ "$#" -ne 3 ]; then
    echo "Usage: . calibration.sh sumSPE30t.py YYMMDD AABB"
    exit 1
fi

PYTHON_SCRIPT=$1
DATE=$2
RUN_ID=$3

echo "Processing files for ${DATE}T${RUN_ID}..."

# Python 스크립트 실행 (YYMMDDTAABB 형식 전달)
python3 $PYTHON_SCRIPT "${DATE}T${RUN_ID}"

echo "Processing complete for ${DATE}T${RUN_ID}"
