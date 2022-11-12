#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 --pty python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 0 --teachers_num 3 --teacher_models resnet34_0 resnet34_1 resnet34_2