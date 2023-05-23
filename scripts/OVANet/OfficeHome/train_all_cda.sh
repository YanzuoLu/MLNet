#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/OVANet/OfficeHome/CDA/art2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/art2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/art2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/clipart2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/clipart2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/clipart2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/product2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/product2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/product2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/real2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/real2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/CDA/real2product.yaml $@