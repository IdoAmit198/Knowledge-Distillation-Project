#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 --pty python3 experiments/hinton_kd.py -d imagewoof -m resnet10 -e 1 -s 42