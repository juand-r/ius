#!/bin/bash

# Batch faithfulness evaluation script
# Runs faithfulness evaluation on all bmds claim extraction directories

set -e  # Exit on any error

echo "Starting batch faithfulness evaluation..."

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Counter for tracking progress
count=0
total=0

# First count total directories
for dir in outputs/summaries-claims/bmds_fixed_size2_8000_all_concat_*_claims_* outputs/summaries-claims/bmds_fixed_size2_8000_all_iterative_*_claims_*; do
    if [[ -d "$dir" ]]; then
        ((total++))
    fi
done

echo "Found $total directories to process"

# Process bmds_fixed_size2_8000_all_concat_* directories
echo "Processing bmds_fixed_size2_8000_all_concat_* directories..."
for dir in outputs/summaries-claims/bmds_fixed_size2_8000_all_concat_*; do
    if [[ -d "$dir" ]]; then
        # Skip specific directory
        if [[ "$dir" == "outputs/summaries-claims/bmds_fixed_size2_8000_all_concat_1e9af3_claims_default-claim-extraction" ]]; then
            echo "Skipping: $dir"
            continue
        fi
        
        ((count++))
        echo "[$count/$total] Processing: $dir"
        
        python -m ius faithfulness \
            --input "$dir" \
            --model gpt-5-mini \
            --method full_text
        
        echo "Completed: $dir"
        echo ""
    fi
done

# Process bmds_fixed_size2_8000_all_iterative_* directories
echo "Processing bmds_fixed_size2_8000_all_iterative_* directories..."
for dir in outputs/summaries-claims/bmds_fixed_size2_8000_all_iterative_*; do
    if [[ -d "$dir" ]]; then
        # Skip specific directory
        if [[ "$dir" == "outputs/summaries-claims/bmds_fixed_size2_8000_all_iterative_94ce59_claims_default-claim-extraction" ]]; then
            echo "Skipping: $dir"
            continue
        fi
        
        ((count++))
        echo "[$count/$total] Processing: $dir"
        
        python -m ius faithfulness \
            --input "$dir" \
            --model gpt-5-mini \
            --method full_text
        
        echo "Completed: $dir"
        echo ""
    fi
done

echo "Batch faithfulness evaluation completed!"
echo "Processed $count directories total"