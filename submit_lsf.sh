#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Change to script directory
cd "$SCRIPT_DIR"

# Run the Python script
echo "Starting feature generation at $(date)"
python test_generate_features.py

echo "Feature generation completed at $(date)"
