#!/bin/bash

# Calculate total OpenAI spending from cumulative-openai-spending.txt
# CSV format: timestamp,model,input_tokens,output_tokens,total_tokens,input_cost,output_cost,total_cost

if [[ ! -f "cumulative-openai-spending.txt" ]]; then
    echo "No spending log found (cumulative-openai-spending.txt)"
    echo "Total: \$0.00"
    exit 0
fi

# Sum the 8th column (total_cost) using awk
total=$(awk -F',' '{sum += $8} END {printf "%.6f", sum}' cumulative-openai-spending.txt)

echo "OpenAI API Spending Summary:"
echo "=========================="
echo "Total entries: $(wc -l < cumulative-openai-spending.txt)"
echo "Total cost: \$${total}"

# Also show last 5 entries for quick overview
echo ""
echo "Last 5 API calls:"
echo "================="
echo "Timestamp,Model,Input,Output,Total,Cost"
tail -5 cumulative-openai-spending.txt | awk -F',' '{printf "%s,%s,%s,%s,%s,$%.6f\n", substr($1,12,8), $2, $3, $4, $5, $8}'