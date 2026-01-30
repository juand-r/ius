#!/bin/bash

# Script to run entity-coverage-multi with --add-reveal for BMDS concat and iterative summary directories
# Usage: ./run_bmds_entity_coverage.sh

echo "🚀 Running entity-coverage-multi --add-reveal for BMDS concat and iterative summary directories..."
echo

# Counter for tracking progress
count=0
total=$(ls -d outputs/summaries/bmds_fixed_size2_8000_all_concat* outputs/summaries/bmds_fixed_size2_8000_all_iterative* 2>/dev/null | wc -l)

if [ $total -eq 0 ]; then
    echo "❌ No matching BMDS summary directories found in outputs/summaries/"
    exit 1
fi

echo "Found $total matching BMDS summary directories to process"
echo "================================================"

# Loop through concat and iterative directories only
for dir in outputs/summaries/bmds_fixed_size2_8000_all_concat* outputs/summaries/bmds_fixed_size2_8000_all_iterative*; do
    if [ -d "$dir" ]; then
        count=$((count + 1))
        echo "[$count/$total] Processing: $(basename "$dir")"
        echo "Running: python -m ius entity-coverage-multi --input $dir --add-reveal"
        
        # Run the command and capture exit status
        if python -m ius entity-coverage-multi --input "$dir" --add-reveal; then
            echo "✅ Success: $(basename "$dir")"
        else
            echo "❌ Failed: $(basename "$dir")"
        fi
        
        echo "================================================"
    fi
done

echo "🎯 Completed processing $count BMDS directories"