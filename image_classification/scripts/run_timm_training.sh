#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/timm_training.py -d imagewoof -m gernet_s -e 50 -s 0 -tt -lr 5e-4