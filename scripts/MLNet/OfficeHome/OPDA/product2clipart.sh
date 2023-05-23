#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/OfficeHome/OPDA/product2clipart.yaml $@