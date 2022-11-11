#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/vit_training.py -d imagewoof -m vit -e 20 -s 0 -tt