#!/bin/bash
python3 -m venv ./venv
source venv/bin/activate
which pip3
# Pytorch 1.10.1 + CUDA 11.3
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install -r requirements.txt