#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/OVANet/Office/ODA/amazon2dslr.yaml $@
python main.py --config-file configs/OVANet/Office/ODA/amazon2webcam.yaml $@
python main.py --config-file configs/OVANet/Office/ODA/dslr2amazon.yaml $@
python main.py --config-file configs/OVANet/Office/ODA/dslr2webcam.yaml $@
python main.py --config-file configs/OVANet/Office/ODA/webcam2amazon.yaml $@
python main.py --config-file configs/OVANet/Office/ODA/webcam2dslr.yaml $@