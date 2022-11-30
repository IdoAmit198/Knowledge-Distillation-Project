#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

# 10 runs with resnet_50 as teacher
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 0 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 1 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 2 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 3 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 4 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 5 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 6 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 7 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 8 -t gernet_s
srun -c 2 --gres=gpu:1 python3 experiments/hinton_kd.py -d imagewoof -m resnet34 -e 100 -s 9 -t gernet_s