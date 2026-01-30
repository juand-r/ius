#!/bin/bash

# Script to run entity-coverage-multi with --add-reveal for true-detective concat and iterative summary directories
# Usage: ./run_true_detective_entity_coverage.sh

echo "🚀 Running entity-coverage-multi --add-reveal for true-detective concat and iterative summary directories..."
echo

# Counter for tracking progress
count=0
total=$(ls -d outputs/summaries/true-detective_fixed_size_2000_all_concat* outputs/summaries/true-detective_fixed_size_2000_all_iterative* 2>/dev/null | wc -l)

if [ $total -eq 0 ]; then
    echo "❌ No matching true-detective summary directories found in outputs/summaries/"
    exit 1
fi

echo "Found $total matching true-detective summary directories to process"
echo "================================================"

# Loop through concat and iterative directories only
for dir in outputs/summaries/true-detective_fixed_size_2000_all_concat* outputs/summaries/true-detective_fixed_size_2000_all_iterative*; do
    if [ -d "$dir" ]; then
        count=$((count + 1))
        echo "[$count/$total] Processing: $(basename "$dir")"
        echo "Running: python -m ius entity-coverage-multi --input $dir --add-reveal"
        
        # Run the command and capture exit status
        if python -m ius entity-coverage-multi --input "$dir" --add-reveal --stop 60; then
            echo "✅ Success: $(basename "$dir")"
        else
            echo "❌ Failed: $(basename "$dir")"
        fi
        
        echo "================================================"
    fi
done

echo "🎯 Completed processing $count true-detective directories"
