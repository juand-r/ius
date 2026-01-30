#!/bin/bash

# Script to run entity coverage evaluation for true-detective summary directories

# Activate virtual environment
source ./venv/bin/activate
source ~/.bashrc

# List of true-detective summary directories
directories=(
    "true-detective_fixed_size_2000_all_concat_1e9af3"
    "true-detective_fixed_size_2000_all_concat_5e8bbe"
    "true-detective_fixed_size_2000_all_concat_65b244"
    "true-detective_fixed_size_2000_all_concat_c6cc9b"
    "true-detective_fixed_size_2000_all_concat_f80fd0"
    "true-detective_fixed_size_2000_all_iterative_061201"
    "true-detective_fixed_size_2000_all_iterative_6a7810"
    "true-detective_fixed_size_2000_all_iterative_9cf072"
    "true-detective_fixed_size_2000_all_iterative_c3c26b"
    "true-detective_fixed_size_2000_all_iterative_c3da60"
)

echo "Running entity coverage evaluation for ${#directories[@]} true-detective directories..."

# Loop through each directory
for dir in "${directories[@]}"; do
    input_dir="outputs/summaries/${dir}/"
    
    # Check if directory exists
    if [ -d "$input_dir" ]; then
        echo ""
        echo "Processing: $dir"
        echo "Command: python -m ius entity-coverage --input $input_dir --range last --model gpt-5-mini --stop 60"
        
        # Run the entity coverage evaluation
        python -m ius entity-coverage --input "$input_dir" --range last --model gpt-5-mini --stop 60
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully processed: $dir"
        else
            echo "❌ Failed to process: $dir"
        fi
    else
        echo "⚠️  Directory not found: $input_dir"
    fi
done

echo ""
echo "Entity coverage evaluation completed for all directories."
