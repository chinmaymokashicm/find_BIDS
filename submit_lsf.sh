#!/bin/bash

# Run the Python script
echo "Starting feature generation at $(date)"
/rsrch5/home/csi/cmokashi/code/find_BIDS/.venv/bin/python test_generate_features.py

echo "Feature generation completed at $(date)"
