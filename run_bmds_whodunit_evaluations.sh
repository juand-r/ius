#!/bin/bash

# Run whodunit evaluations for all bmds concat and iterative summaries
# Usage: ./run_bmds_whodunit_evaluations.sh

SUMMARIES_DIR="outputs/summaries"

echo "Running whodunit evaluations for bmds concat and iterative summaries..."

# Process concat directories
echo "Processing concat directories..."
for dir in "$SUMMARIES_DIR"/bmds_fixed_size2_8000_all_concat_*; do
    if [ -d "$dir" ]; then
        echo "Processing: $(basename $dir)"
        python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input "$dir/" --scoring-prompt two-calls
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed: $(basename $dir)"
        else
            echo "❌ Failed to process: $(basename $dir)"
        fi
        echo "---"
    fi
done

# Process iterative directories  
echo "Processing iterative directories..."
for dir in "$SUMMARIES_DIR"/bmds_fixed_size2_8000_all_iterative_*; do
    if [ -d "$dir" ]; then
        echo "Processing: $(basename $dir)"
        python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input "$dir/" --scoring-prompt two-calls
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed: $(basename $dir)"
        else
            echo "❌ Failed to process: $(basename $dir)"
        fi
        echo "---"
    fi
done

echo "Finished processing all bmds directories."
