#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate MLNet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/OfficeHome/ODA/art2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/art2product.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/art2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/clipart2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/clipart2product.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/clipart2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/product2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/product2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/product2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/real2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/real2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/ODA/real2product.yaml $@