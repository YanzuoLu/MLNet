#!/bin/bash

ulimit -n 65536
source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/OfficeHome/CDA/art2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/art2product.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/art2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/clipart2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/clipart2product.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/clipart2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/product2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/product2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/product2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/real2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/real2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/CDA/real2product.yaml $@