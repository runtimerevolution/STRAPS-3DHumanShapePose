#!/bin/bash

# Install Pillow dependencies and Ninja for detectron2 build
sudo apt install zlib1g-dev libpng-dev libtiff-dev libfreetype6-dev ninja-build

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

pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install 'git+https://github.com/facebookresearch/detectron2.git'
