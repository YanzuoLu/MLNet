#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/OfficeHome/PDA/art2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/art2product.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/art2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/clipart2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/clipart2product.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/clipart2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/product2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/product2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/product2real.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/real2art.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/real2clipart.yaml $@
python main.py --config-file configs/MLNet/OfficeHome/PDA/real2product.yaml $@