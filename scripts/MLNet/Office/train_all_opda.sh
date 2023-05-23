#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/Office/OPDA/amazon2dslr.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/amazon2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/dslr2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/dslr2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/webcam2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/webcam2dslr.yaml $@