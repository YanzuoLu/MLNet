#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/Office/PDA/amazon2dslr.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/amazon2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/dslr2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/dslr2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/webcam2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/webcam2dslr.yaml $@