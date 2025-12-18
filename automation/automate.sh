#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="./decompose.py"

# Define arguments
PROMPT_PATH="./data/malicious_targets.csv"
MODEL="gpt-4o-mini"
GENERATE_MODE="joint"
SAVE_PATH="./attack_prompt_data/automated_processing_results/decompose.json"
OFFSET=0
TOTAL_NUMBER=10

# Execute the Python script with the arguments
python $PYTHON_SCRIPT --prompt_path $PROMPT_PATH --model $MODEL --generate_mode $GENERATE_MODE --save_path $SAVE_PATH --offset $OFFSET --total_number $TOTAL_NUMBER