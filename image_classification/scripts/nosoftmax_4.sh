#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 0 -t resnet50 -ud

srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 1 -t resnet50 -ud
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 2 -t resnet50 -ud
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 3 -t resnet50 -ud




