#!/bin/bash

# Script to run faithfulness evaluation on all claim extraction directories
# Usage: ./run_faithfulness_evaluation_all.sh [dataset1] [dataset2] ...
# Examples:
#   ./run_faithfulness_evaluation_all.sh                    # Default: bmds, detectiveqa
#   ./run_faithfulness_evaluation_all.sh true-detective     # Process only true-detective
#   ./run_faithfulness_evaluation_all.sh bmds true-detective # Process bmds and true-detective

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to log success
log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

# Function to log error
log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

# Function to log warning
log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "outputs/summaries-claims" ]; then
    log_error "outputs/summaries-claims directory not found. Please run this script from the project root."
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    log "Activating virtual environment..."
    source venv/bin/activate
fi

# Initialize counters
total_processed=0
successful=0
failed=0

# Function to run faithfulness evaluation
run_faithfulness() {
    local dir=$1
    local dataset_type=$2
    
    log "Processing ${dataset_type}: ${dir}"
    
    if python -m ius faithfulness \
        --input "outputs/summaries-claims/${dir}" \
        --model gpt-4o \
        --range last \
        --claim-stop 50 \
        --method full_text \
        --add-reveal; then
        log_success "Completed: ${dir}"
        ((successful++))
    else
        log_error "Failed: ${dir}"
        ((failed++))
    fi
    
    ((total_processed++))
    echo ""
}

# Function to process a dataset
process_dataset() {
    local dataset_name=$1
    local display_name=$2
    
    log "Starting ${display_name} faithfulness evaluations..."
    echo "=================================================="
    
    local dirs=$(ls outputs/summaries-claims/ 2>/dev/null | grep "^${dataset_name}_" | sort || true)
    local count=0
    
    # Count directories
    if [ -n "$dirs" ]; then
        count=$(echo "$dirs" | wc -l | tr -d ' ')
    fi
    
    if [ "$count" -eq 0 ]; then
        log_warning "No ${display_name} directories found, skipping..."
        echo ""
        return
    fi
    
    log "Found ${count} ${display_name} directories to process"
    
    for dir in $dirs; do
        run_faithfulness "$dir" "$display_name"
    done
    
    log_success "Completed all ${display_name} directories (${count} processed)"
    echo ""
}

# Determine which datasets to process
if [ $# -eq 0 ]; then
    # Default behavior: process bmds and detectiveqa (backward compatible)
    datasets=("bmds:BMDS" "detectiveqa:DetectiveQA")
else
    # Process specified datasets
    datasets=()
    for arg in "$@"; do
        case "$arg" in
            bmds)
                datasets+=("bmds:BMDS")
                ;;
            detectiveqa)
                datasets+=("detectiveqa:DetectiveQA")
                ;;
            true-detective)
                datasets+=("true-detective:True-Detective")
                ;;
            squality)
                datasets+=("squality:SQuality")
                ;;
            *)
                log_error "Unknown dataset: $arg"
                log "Supported datasets: bmds, detectiveqa, true-detective, squality"
                exit 1
                ;;
        esac
    done
fi

# Process each dataset
for dataset_pair in "${datasets[@]}"; do
    dataset_name="${dataset_pair%%:*}"
    display_name="${dataset_pair##*:}"
    process_dataset "$dataset_name" "$display_name"
done

# Final summary
echo "=================================================="
log "FINAL SUMMARY"
echo "=================================================="
log "Total directories processed: ${total_processed}"
log_success "Successful: ${successful}"
if [ $failed -gt 0 ]; then
    log_error "Failed: ${failed}"
else
    log_success "Failed: ${failed}"
fi

if [ $failed -eq 0 ]; then
    log_success "All faithfulness evaluations completed successfully! 🎉"
else
    log_warning "Some evaluations failed. Check the logs above for details."
    exit 1
fi