#!/usr/bin/env bash
set -euo pipefail
INPUT="supert_toy.jsonl"

# One-time setup (uncomment if you haven't set up SUPERT yet)
# conda env create -f environments/supert.yml -n supert
# export SUPERT_ENV=supert
# sacrerouge setup-metric supert

# Run SUPERT (document-based reader)
python -m sacrerouge supert evaluate       --input-files "$INPUT"       --dataset-reader document-based       --macro-output-json supert_macro.json       --micro-output-jsonl supert_micro.jsonl
