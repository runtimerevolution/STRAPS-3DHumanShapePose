#!/bin/bash

# Set a custom temp directory to avoid "No space left on device" errors
export TMPDIR=/opt/dlami/nvme/straps_tmp

# Ensure the temp directory exists
mkdir -p "$TMPDIR"

# Install each requirement with --no-build-isolation
while read -r req; do
    # Skip empty lines and comments
    if [[ -n "$req" && ! "$req" =~ ^# ]]; then
        echo "Installing $req..."
        pip install --no-build-isolation "$req"
    fi
done < requirements.txt
