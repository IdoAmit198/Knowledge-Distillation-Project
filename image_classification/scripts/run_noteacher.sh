#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 --pty python3 experiments/no_teacher.py -d imagewoof -m resnet18 -e 100 -s 42