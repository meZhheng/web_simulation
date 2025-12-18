@echo off

REM Define the path to the Python script
set PYTHON_SCRIPT=decompose.py

REM Define arguments
set PROMPT_PATH=.\data\malicious_targets.csv
set MODEL=gpt-4o-mini
set GENERATE_MODE=joint
set SAVE_PATH=.\attack_prompt_data\automated_processing_results\decompose.json
set OFFSET=0
set TOTAL_NUMBER=10

REM Execute the Python script with the arguments
python %PYTHON_SCRIPT% ^
  --prompt_path %PROMPT_PATH% ^
  --model %MODEL% ^
  --generate_mode %GENERATE_MODE% ^
  --save_path %SAVE_PATH% ^
  --offset %OFFSET% ^
  --total_number %TOTAL_NUMBER%
