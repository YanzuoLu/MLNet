#!/bin/bash

ulimit -n 65536
source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/OVANet/OfficeHome/PDA/art2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/art2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/art2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/clipart2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/clipart2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/clipart2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/product2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/product2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/product2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/real2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/real2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/PDA/real2product.yaml $@