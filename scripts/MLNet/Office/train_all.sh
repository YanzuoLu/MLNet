#!/bin/bash

ulimit -n 65536
source $(dirname "${CONDA_PYTHON_EXE}")/activate mlnet
export OMP_NUM_THREADS=$(($(nproc --all)/$(nvidia-smi -L | wc -l)))
export MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=$1
shift

python main.py --config-file configs/MLNet/Office/CDA/amazon2dslr.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/amazon2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/dslr2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/dslr2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/webcam2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/CDA/webcam2dslr.yaml $@

python main.py --config-file configs/MLNet/Office/ODA/amazon2dslr.yaml $@
python main.py --config-file configs/MLNet/Office/ODA/amazon2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/ODA/dslr2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/ODA/dslr2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/ODA/webcam2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/ODA/webcam2dslr.yaml $@

python main.py --config-file configs/MLNet/Office/OPDA/amazon2dslr.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/amazon2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/dslr2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/dslr2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/webcam2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/OPDA/webcam2dslr.yaml $@

python main.py --config-file configs/MLNet/Office/PDA/amazon2dslr.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/amazon2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/dslr2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/dslr2webcam.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/webcam2amazon.yaml $@
python main.py --config-file configs/MLNet/Office/PDA/webcam2dslr.yaml $@