#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/OVANet/OfficeHome/ODA/art2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/art2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/art2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/clipart2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/clipart2product.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/clipart2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/product2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/product2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/product2real.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/real2art.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/real2clipart.yaml $@
python main.py --config-file configs/OVANet/OfficeHome/ODA/real2product.yaml $@