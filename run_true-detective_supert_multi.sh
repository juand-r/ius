#!/bin/bash

# Run overlap-metrics-multi SUPERT evaluation on True-Detective summary collections
# Processes all true-detective_fixed_size_2000_all_concat_* and true-detective_fixed_size_2000_all_iterative_* directories

set -e  # Exit on any error

echo "Starting True-Detective overlap-metrics-multi SUPERT evaluation..."
echo "Target patterns: true-detective_fixed_size_2000_all_concat_* and true-detective_fixed_size_2000_all_iterative_*"
echo "=================================================================="

# Counter for tracking progress
total_dirs=0
processed_dirs=0
failed_dirs=0

# Find all matching directories (excluding BAD- prefixed ones)
summary_dirs=$(ls -1 outputs/summaries/ | grep -E "^true-detective_fixed_size_2000_all_(concat|iterative)_" | grep -v "^BAD-")

# Count total directories
total_dirs=$(echo "$summary_dirs" | wc -l)
echo "Found $total_dirs directories to process"
echo

# Process each directory
for dir in $summary_dirs; do
    processed_dirs=$((processed_dirs + 1))
    echo "[$processed_dirs/$total_dirs] Processing: $dir"
    echo "Command: python -m ius overlap-metrics-multi --supert --input outputs/summaries/$dir --add-reveal --stop 30"
    
    if python -m ius overlap-metrics-multi --supert --input "outputs/summaries/$dir" --add-reveal --stop 30; then
        echo "✓ SUCCESS: $dir"
    else
        echo "✗ FAILED: $dir"
        failed_dirs=$((failed_dirs + 1))
    fi
    echo "=================================================================="
done

echo "True-Detective overlap-metrics-multi SUPERT evaluation completed!"
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
