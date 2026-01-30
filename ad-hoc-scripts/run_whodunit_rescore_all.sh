#!/bin/bash

# Script to recreate whodunit evaluations with updated two-calls scoring prompts
# Based on extracted commands from existing evaluation directories

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Whodunit Rescore Script - Running All Evaluations with Two-Calls Scoring${NC}"
echo "=========================================================================="

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Virtual environment not detected${NC}"
    echo "Attempting to activate ./venv/bin/activate..."
    if [[ -f "./venv/bin/activate" ]]; then
        source ./venv/bin/activate
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    else
        echo -e "${RED}❌ Error: ./venv/bin/activate not found${NC}"
        exit 1
    fi
fi

# Source bashrc for OpenAI API key
if [[ -f ~/.bashrc ]]; then
    echo "Sourcing ~/.bashrc for OpenAI API key..."
    source ~/.bashrc
fi

# Check if OpenAI API key is set
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo -e "${RED}❌ Error: OPENAI_API_KEY not set${NC}"
    echo "Please set your OpenAI API key in ~/.bashrc"
    exit 1
fi

# Create log directory
mkdir -p logs
log_file="logs/whodunit_rescore_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

# Array of commands to run (extracted and modified from collection.json files)
commands=(
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_concat_131eac/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_concat_5e8bbe/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_concat_65b244/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_concat_c6cc9b/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_iterative_061201/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_iterative_94ce59/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_iterative_9cf072/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range penultimate --input outputs/summaries/bmds_fixed_size2_8000_all_iterative_c3c26b/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices-reveal --model o3 --range all --input outputs/chunks/bmds_fixed_size2_8000/ --scoring-prompt two-calls --rescore"
    "python -m ius whodunit --prompt default-whodunit-culprits-and-accomplices --model o3 --range all --input outputs/chunks/bmds_fixed_size2_8000/ --scoring-prompt two-calls --rescore"
)

# Command descriptions for logging
descriptions=(
    "bmds_fixed_size2_8000_all_concat_131eac (penultimate summaries)"
    "bmds_fixed_size2_8000_all_concat_5e8bbe (penultimate summaries)"
    "bmds_fixed_size2_8000_all_concat_65b244 (penultimate summaries)"
    "bmds_fixed_size2_8000_all_concat_c6cc9b (penultimate summaries)"
    "bmds_fixed_size2_8000_all_iterative_061201 (penultimate summaries)"
    "bmds_fixed_size2_8000_all_iterative_94ce59 (penultimate summaries)"
    "bmds_fixed_size2_8000_all_iterative_9cf072 (penultimate summaries)"
    "bmds_fixed_size2_8000_all_iterative_c3c26b (penultimate summaries)"
    "bmds_fixed_size2_8000_whodunit_481302 (all chunks with reveal prompt)"
    "bmds_fixed_size2_8000_whodunit_5964ea (all chunks)"
)

echo -e "${BLUE}📋 Will run ${#commands[@]} evaluation commands:${NC}"
for i in "${!descriptions[@]}"; do
    echo "  $((i+1)). ${descriptions[i]}"
done

echo ""
echo -e "${BLUE}🚀 Starting evaluation process...${NC}"
echo ""

# Start processing
total_commands=${#commands[@]}
start_time=$(date +%s)

for i in "${!commands[@]}"; do
    current_command=$((i + 1))
    cmd="${commands[i]}"
    desc="${descriptions[i]}"
    
    log_with_timestamp "${BLUE}📂 Running command $current_command/$total_commands: $desc${NC}"
    log_with_timestamp "Command: $cmd"
    
    # Run the command
    if eval "$cmd" 2>&1 | tee -a "$log_file"; then
        log_with_timestamp "${GREEN}✅ Successfully completed: $desc${NC}"
    else
        log_with_timestamp "${RED}❌ Error running: $desc${NC}"
        log_with_timestamp "Continuing with next command..."
    fi
    
    # Calculate and show progress
    elapsed=$(($(date +%s) - start_time))
    if [[ $current_command -gt 0 ]]; then
        avg_time_per_cmd=$((elapsed / current_command))
        remaining_cmds=$((total_commands - current_command))
        estimated_remaining=$((avg_time_per_cmd * remaining_cmds))
        
        log_with_timestamp "${YELLOW}📊 Progress: $current_command/$total_commands completed${NC}"
        log_with_timestamp "Elapsed: ${elapsed}s, Avg per cmd: ${avg_time_per_cmd}s, Est. remaining: ${estimated_remaining}s"
    fi
    echo ""
done

# Final summary
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo ""
echo "=========================================================================="
log_with_timestamp "${GREEN}🎉 All whodunit evaluations completed!${NC}"
log_with_timestamp "Total commands run: $total_commands"
log_with_timestamp "Total time: ${hours}h ${minutes}m ${seconds}s"
log_with_timestamp "Log file: $log_file"
echo "=========================================================================="

echo -e "${BLUE}💡 Next steps:${NC}"
echo "1. Check the log file for any errors: $log_file"
echo "2. Compare results with your FIXED version using comparison scripts"
echo "3. Analyze improvements from the updated two-calls scoring prompts"
echo "4. All evaluations now use consistent scoring with fixed double-nesting structure"
