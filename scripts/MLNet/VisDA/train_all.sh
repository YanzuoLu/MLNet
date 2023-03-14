#!/bin/bash

ulimit -n 65536
source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/VisDA/CDA/default.yaml $@
python main.py --config-file configs/MLNet/VisDA/ODA/default.yaml $@
python main.py --config-file configs/MLNet/VisDA/OPDA/default.yaml $@
python main.py --config-file configs/MLNet/VisDA/PDA/default.yaml $@