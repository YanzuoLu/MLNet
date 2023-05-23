#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/OVANet/OfficeHome/OPDA/art2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/art2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/art2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/clipart2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/clipart2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/clipart2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/product2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/product2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/product2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/real2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/real2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/OPDA/real2product.yaml $@