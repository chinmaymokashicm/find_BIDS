#!/bin/bash
#BSUB -J find_bids_features
#BSUB -q medium
#BSUB -n 8
#BSUB -M 16000
#BSUB -W 12:00
#BSUB -o /rsrch5/home/csi/Quarles_Lab/find_BIDS/logs/find_bids_features.%J.out
#BSUB -e /rsrch5/home/csi/Quarles_Lab/find_BIDS/logs/find_bids_features.%J.err
#BSUB -R "span[ptile=8]"

# Exit on error
set -e

# Load modules if needed (uncomment if required)
# module load python/3.11

# Get the directory where this script is located
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
