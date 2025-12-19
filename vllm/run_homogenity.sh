#!/bin/bash
# set -euo pipefail

IP="127.0.0.1"
PORT="8080"

FILE_PATH="/home/liyong/code/CityHomogeneity/output/df_top1_top50_removed_highway.csv"
OUTPUT_PATH="/home/liyong/code/CityHomogeneity/output/qwen_cls_test.jsonl"
# Optional: set BASE_PATH if image paths in the CSV are relative.
BASE_PATH="/data_ssd/"
cd /data_nas/liyong/code/vllm

if [ -f "$FILE_PATH" ]; then
    echo "Processing file: $FILE_PATH"
    if [ -n "$BASE_PATH" ]; then
        python3 homogenity_cls.py "$FILE_PATH" "$OUTPUT_PATH" "$IP" "$PORT" --base_path "$BASE_PATH"
    else
        python3 homogenity_cls.py "$FILE_PATH" "$OUTPUT_PATH" "$IP" "$PORT"
    fi
else
    echo "Warning: File not found: $FILE_PATH"
fi

echo "Processing finished."
