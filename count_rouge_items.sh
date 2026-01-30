#!/bin/bash

echo "Counting files in items/ directories under outputs/eval/intrinsic/rouge-continuity/"
echo "=============================================================="

# Check if the rouge-continuity directory exists
if [ ! -d "outputs/eval/intrinsic/rouge-continuity" ]; then
    echo "Error: outputs/eval/intrinsic/rouge-continuity directory not found!"
    exit 1
fi

# Loop through all directories in outputs/eval/intrinsic/rouge-continuity/
for dir in outputs/eval/intrinsic/rouge-continuity/*/; do
    if [ -d "$dir" ]; then
        items_dir="${dir}items/"
        if [ -d "$items_dir" ]; then
            # Count files in the items directory
            file_count=$(find "$items_dir" -maxdepth 1 -type f | wc -l)
            dir_name=$(basename "$dir")
            printf "%-50s %s files\n" "$dir_name" "$file_count"
        else
            dir_name=$(basename "$dir")
            printf "%-50s (no items/ directory)\n" "$dir_name"
        fi
    fi
done

echo ""
echo "Total directories checked: $(find outputs/eval/intrinsic/rouge-continuity/ -maxdepth 1 -type d | tail -n +2 | wc -l)"
