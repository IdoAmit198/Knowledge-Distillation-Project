#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

# 10 runs with resnet_34 as teacher
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 40 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 41 -t alexnet 
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 42 -t alexnet 
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 43 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 44 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 45 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 46 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 47 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 48 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 49 -t alexnet

srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 40 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 41 -t alexnet 
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 42 -t alexnet 
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 43 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 44 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 45 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 46 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 47 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 48 -t alexnet
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet18 -e 100 -s 49 -t alexnet