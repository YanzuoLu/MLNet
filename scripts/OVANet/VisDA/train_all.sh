#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/OVANet/VisDA/CDA/default.yaml $@
python main.py --config-file configs/OVANet/VisDA/ODA/default.yaml $@
python main.py --config-file configs/OVANet/VisDA/OPDA/default.yaml $@
python main.py --config-file configs/OVANet/VisDA/PDA/default.yaml $@