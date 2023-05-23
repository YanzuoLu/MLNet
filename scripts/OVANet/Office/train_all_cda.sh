#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/OVANet/Office/CDA/amazon2dslr.yaml $@
python main.py --config-file configs/OVANet/Office/CDA/amazon2webcam.yaml $@
python main.py --config-file configs/OVANet/Office/CDA/dslr2amazon.yaml $@
python main.py --config-file configs/OVANet/Office/CDA/dslr2webcam.yaml $@
python main.py --config-file configs/OVANet/Office/CDA/webcam2amazon.yaml $@
python main.py --config-file configs/OVANet/Office/CDA/webcam2dslr.yaml $@