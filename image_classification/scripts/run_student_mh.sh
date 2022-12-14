#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/student_MH_kd.py -d imagewoof -m resnet18 -e 2 -s 121 \
--teachers_num 1 --teacher_models resnet34_1 resnet101
# -t resnet101