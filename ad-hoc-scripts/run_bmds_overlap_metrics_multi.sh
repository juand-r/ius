#!/bin/bash

# Run overlap-metrics-multi ROUGE evaluation on BMDS summary collections
# Processes all bmds_fixed_size2_8000_all_concat_* and bmds_fixed_size2_8000_all_iterative_* directories

set -e  # Exit on any error

echo "Starting BMDS overlap-metrics-multi ROUGE evaluation..."
echo "Target patterns: bmds_fixed_size2_8000_all_concat_* and bmds_fixed_size2_8000_all_iterative_*"
echo "=================================================================="

# Counter for tracking progress
total_dirs=0
processed_dirs=0
failed_dirs=0

# Find all matching directories (excluding BAD- prefixed ones)
summary_dirs=$(ls -1 outputs/summaries/ | grep -E "^bmds_fixed_size2_8000_all_(concat|iterative)_" | grep -v "^BAD-")

# Count total directories
total_dirs=$(echo "$summary_dirs" | wc -l)
echo "Found $total_dirs directories to process"
echo

# Process each directory
for dir in $summary_dirs; do
    processed_dirs=$((processed_dirs + 1))
    echo "[$processed_dirs/$total_dirs] Processing: $dir"
    echo "Command: python -m ius overlap-metrics-multi --rouge --input outputs/summaries/$dir --add-reveal"
    
    if python -m ius overlap-metrics-multi --rouge --input "outputs/summaries/$dir" --add-reveal; then
        echo "✓ SUCCESS: $dir"
    else
        echo "✗ FAILED: $dir"
        failed_dirs=$((failed_dirs + 1))
    fi
    echo "=================================================================="
done

echo "BMDS overlap-metrics-multi evaluation completed!"
echo "Total directories: $total_dirs"
echo "Successfully processed: $((processed_dirs - failed_dirs))"
echo "Failed: $failed_dirs"

if [ $failed_dirs -eq 0 ]; then
    echo "🎉 All evaluations completed successfully!"
    exit 0
else
    echo "⚠️  Some evaluations failed. Check the output above for details."
    exit 1
fi
