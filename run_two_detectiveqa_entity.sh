#!/bin/bash

# Script to run entity-coverage-multi for all detectiveqa summary directories
# Using --add-reveal flag to include reveal text in entity coverage evaluation

set -e  # Exit on any error

echo "🔍 Running entity coverage evaluation for all detectiveqa directories..."

# CONCAT methods
echo "📊 Processing CONCAT methods..."
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_concat_131eac/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_concat_1bfbd2/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_concat_5e8bbe/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_concat_65b244/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_concat_c6cc9b/ --add-reveal

echo ""
echo "🔄 Processing ITERATIVE methods..."
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_iterative_061201/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_iterative_8c494b/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_iterative_94ce59/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_iterative_9cf072/ --add-reveal
python -m ius entity-coverage-multi --input outputs/summaries/detectiveqa_fixed_size_80000_all_iterative_c3c26b/ --add-reveal

echo ""
echo "✅ All detectiveqa entity coverage evaluations completed!"
