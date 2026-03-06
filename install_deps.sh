#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
# Install CPU wheels for PyTorch to avoid pulling heavy CUDA artifacts by default.
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
# Install remaining dependencies.
python -m pip install -r requirements.txt
