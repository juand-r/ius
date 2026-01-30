#!/bin/bash

# Run whodunit evaluations for all true-detective summaries
# Usage: ./run_true_detective_whodunit_evaluations.sh

SUMMARIES_DIR="outputs/summaries"

echo "Running whodunit evaluations for true-detective summaries..."

for dir in "$SUMMARIES_DIR"/true-detective*; do
    if [ -d "$dir" ]; then
        echo "Processing: $(basename $dir)"
        python -m ius --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input "$dir/" --scoring-prompt two-calls
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed: $(basename $dir)"
        else
            echo "❌ Failed to process: $(basename $dir)"
        fi
        echo "---"
    fi
done

echo "Finished processing all true-detective directories."