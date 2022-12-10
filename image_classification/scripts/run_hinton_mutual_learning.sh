#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 16 -ml -mnn 1 -t resnet50