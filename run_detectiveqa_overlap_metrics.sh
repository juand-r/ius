#!/bin/bash

# Script to run overlap-metrics-multi for all detectiveqa and true-detective summary directories
# Usage: ./run_detectiveqa_overlap_metrics.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Finding detectiveqa and true-detective summary directories...${NC}"

# Find all detectiveqa directories in outputs/summaries/
detectiveqa_dirs=(outputs/summaries/detectiveqa_*)

# Find all true-detective directories in outputs/summaries/
true_detective_dirs=(outputs/summaries/true-detective_*)

# Check if any detectiveqa directories were found
if [ ${#detectiveqa_dirs[@]} -eq 0 ] || [ ! -d "${detectiveqa_dirs[0]}" ]; then
    echo -e "${RED}❌ No detectiveqa directories found in outputs/summaries/${NC}"
    echo -e "${YELLOW}Expected pattern: outputs/summaries/detectiveqa_*${NC}"
    detectiveqa_dirs=()
fi

# Check if any true-detective directories were found
if [ ${#true_detective_dirs[@]} -eq 0 ] || [ ! -d "${true_detective_dirs[0]}" ]; then
    echo -e "${RED}❌ No true-detective directories found in outputs/summaries/${NC}"
    echo -e "${YELLOW}Expected pattern: outputs/summaries/true-detective_*${NC}"
    true_detective_dirs=()
fi

# Exit if no directories found at all
if [ ${#detectiveqa_dirs[@]} -eq 0 ] && [ ${#true_detective_dirs[@]} -eq 0 ]; then
    echo -e "${RED}❌ No directories found to process${NC}"
    exit 1
fi

if [ ${#detectiveqa_dirs[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Found ${#detectiveqa_dirs[@]} detectiveqa directories:${NC}"
    for dir in "${detectiveqa_dirs[@]}"; do
        echo -e "  📁 $(basename "$dir")"
    done
fi

if [ ${#true_detective_dirs[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Found ${#true_detective_dirs[@]} true-detective directories:${NC}"
    for dir in "${true_detective_dirs[@]}"; do
        echo -e "  📁 $(basename "$dir")"
    done
fi

echo ""
echo -e "${BLUE}🚀 Starting overlap metrics evaluation...${NC}"

# Counter for tracking progress
total=$((${#detectiveqa_dirs[@]} + ${#true_detective_dirs[@]}))
current=0
successful=0
failed=0

# Function to process a directory
process_directory() {
    local dir=$1
    local dataset_type=$2
    
    current=$((current + 1))
    dir_name=$(basename "$dir")
    
    echo ""
    echo -e "${BLUE}[${current}/${total}] Processing ${dataset_type}: ${dir_name}${NC}"
    echo -e "${YELLOW}Running: python -m ius overlap-metrics-multi --input $dir --rouge${NC}"
    
    # Run the command and capture exit status
    if python -m ius overlap-metrics-multi --input "$dir" --rouge; then
        echo -e "${GREEN}✅ Successfully processed: ${dir_name}${NC}"
        successful=$((successful + 1))
    else
        echo -e "${RED}❌ Failed to process: ${dir_name}${NC}"
        failed=$((failed + 1))
    fi
}

# Process detectiveqa directories first
if [ ${#detectiveqa_dirs[@]} -gt 0 ]; then
    echo ""
    echo -e "${BLUE}📋 Processing DetectiveQA directories (${#detectiveqa_dirs[@]} found)...${NC}"
    for dir in "${detectiveqa_dirs[@]}"; do
        process_directory "$dir" "DetectiveQA"
    done
fi

# Process true-detective directories second
if [ ${#true_detective_dirs[@]} -gt 0 ]; then
    echo ""
    echo -e "${BLUE}📋 Processing True-Detective directories (${#true_detective_dirs[@]} found)...${NC}"
    for dir in "${true_detective_dirs[@]}"; do
        process_directory "$dir" "True-Detective"
    done
fi

echo ""
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "${GREEN}  ✅ Successful: ${successful}/${total}${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}  ❌ Failed: ${failed}/${total}${NC}"
fi
echo -e "${BLUE}  📁 Total directories processed: ${total}${NC}"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All detectiveqa and true-detective overlap metrics completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some directories failed processing. Check logs above for details.${NC}"
    exit 1
fi