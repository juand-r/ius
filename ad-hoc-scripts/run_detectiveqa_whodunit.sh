#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Base directory containing the summary outputs
SUMMARIES_DIR="outputs/summaries"

# Find all directories starting with "detectiveqa_"
for dir in "$SUMMARIES_DIR"/detectiveqa_fixed_size_80000_*; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        python -m ius whodunit \
            --prompt detectiveqa-whodunit-culprits-and-accomplices \
            --model o3 \
            --range penultimate \
            --input "$dir" \
            --scoring-prompt whodunit-scoring-culprits-detectiveqa
        echo "Completed processing: $dir"
        echo "---"
    fi
done

echo "All detectiveqa directories processed."
