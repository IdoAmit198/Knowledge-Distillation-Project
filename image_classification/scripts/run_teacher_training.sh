#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 --pty python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 20 -s 0 -tt