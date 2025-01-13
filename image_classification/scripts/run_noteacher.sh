#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 1 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 2 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 3 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 4 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 5 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 6 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 7 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 8 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 9 --teacher_training --no_pre_trained_teacher
srun -c 2 --gres=gpu:1 python3 experiments/no_teacher.py -d imagewoof -m resnet50 -e 100 -s 10 --teacher_training --no_pre_trained_teacher