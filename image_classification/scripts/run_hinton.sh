#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 16 -t vit