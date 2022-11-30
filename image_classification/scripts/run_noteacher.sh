#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m alexnet -e 20 -s 0 --teacher_training -lr 5e-5