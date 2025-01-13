#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

# 10 runs with resnet_101 as teacher
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 80 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 81 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 82 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 83 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 84 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 85 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 86 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 87 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 88 -t resnet101
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 89 -t resnet101