#!/bin/bash

# Step 1: Create a new conda environment with Python 3.10
conda create -n llava python=3.10 -y

# Step 2: Activate the conda environment
conda activate llava

# Step 3: Upgrade pip to the latest version to support PEP 660
pip install --upgrade pip

# Step 4: Install the package in editable mode with train extras
pip install -e ".[train]"

# Step 5: Install specific versions of required packages
pip install pynvml==11.5.0
pip install accelerate==0.29.3
pip install flash-attn==2.5.7

echo "All packages installed successfully in the llava environment!"
