#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd_fc.py -d imagewoof -m resnet18 -e 1 -s 16 -t resnet50