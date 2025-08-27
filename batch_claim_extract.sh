#!/bin/bash

echo "Starting batch claim extraction for BMDS datasets..."
echo "================================================================"

# Counter for tracking progress
total_dirs=0
processed_dirs=0
failed_dirs=0

# Count total directories first
for dir in outputs/summaries/detectiveqa_fixed_size_80000_* outputs/summaries/true-detective_fixed_size_2000_*; do
    if [ -d "$dir" ]; then
        ((total_dirs++))
    fi
done

echo "Found $total_dirs directories to process"
echo ""

# Process each directory
for dir in outputs/summaries/detectiveqa_fixed_size_80000_* outputs/summaries/true-detective_fixed_size_2000_*; do
    if [ -d "$dir" ]; then
        ((processed_dirs++))
        echo "[$processed_dirs/$total_dirs] Processing: $dir"
        
        # Run claim extraction
        if python -m ius.cli.claim_extract --input "$dir" --model gpt-5-mini --stop 60; then
            echo "✓ Successfully processed: $dir"
        else
            echo "✗ Failed to process: $dir"
            ((failed_dirs++))
        fi
        
        echo ""
    fi
done

echo "================================================================"
echo "BATCH CLAIM EXTRACTION COMPLETED"
echo "================================================================"
echo "Total directories: $total_dirs"
echo "Successfully processed: $((processed_dirs - failed_dirs))"
echo "Failed: $failed_dirs"
