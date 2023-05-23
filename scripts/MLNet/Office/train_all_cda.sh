#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/Office/CDA/amazon2dslr.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/amazon2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/dslr2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/dslr2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/webcam2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/webcam2dslr.yaml $@