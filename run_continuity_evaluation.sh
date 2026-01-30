#!/bin/bash

# Script to run continuity evaluation for summary directories in specified order
# Usage: ./run_continuity_evaluation.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Running ROUGE continuity evaluation in specified order...${NC}"

# Define patterns in the requested order
patterns=(
    "bmds_fixed_size2_8000_all_concat*"
    "bmds_fixed_size2_8000_all_iterative*" 
    "detectiveqa_fixed_size_80000_all*"
    "true-detective_fixed_size_2000_all*"
)

# Counter for tracking progress
total_processed=0
successful=0
failed=0

for pattern in "${patterns[@]}"; do
    echo ""
    echo -e "${YELLOW}📂 Processing pattern: ${pattern}${NC}"
    
    # Find directories matching the pattern
    dirs=(outputs/summaries/${pattern})
    
    # Check if any directories were found (and they actually exist)
    pattern_dirs=()
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            pattern_dirs+=("$dir")
        fi
    done
    
    if [ ${#pattern_dirs[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No directories found for pattern: ${pattern}${NC}"
        continue
    fi
    
    echo -e "${GREEN}✅ Found ${#pattern_dirs[@]} directories for pattern: ${pattern}${NC}"
    
    # Sort directories for consistent ordering
    IFS=$'\n' sorted_dirs=($(sort <<<"${pattern_dirs[*]}"))
    unset IFS
    
    # Process each directory
    for dir in "${sorted_dirs[@]}"; do
        total_processed=$((total_processed + 1))
        dir_name=$(basename "$dir")
        
        echo ""
        echo -e "${BLUE}[${total_processed}] Processing: ${dir_name}${NC}"
        echo -e "${YELLOW}Running: python -m ius continuity --input $dir --rouge --range all${NC}"
        
        # Run the continuity evaluation
        if python -m ius continuity --input "$dir" --rouge --range all; then
            echo -e "${GREEN}✅ Successfully processed: ${dir_name}${NC}"
            successful=$((successful + 1))
        else
            echo -e "${RED}❌ Failed to process: ${dir_name}${NC}"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "${GREEN}  ✅ Successful: ${successful}/${total_processed}${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}  ❌ Failed: ${failed}/${total_processed}${NC}"
fi
echo -e "${BLUE}  📁 Total directories processed: ${total_processed}${NC}"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All continuity evaluations completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some directories failed processing. Check logs above for details.${NC}"
    exit 1
fi