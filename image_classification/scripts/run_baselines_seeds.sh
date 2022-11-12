#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate skd

for i in {4..9}
do
    sbatch -c 2 --gres=gpu:1 -o output_files/baselines/no_teacher_baseline_seed_$i.out -J no_t$i scripts/run_noteacher_seed.sh $i
done

for i in {5..9}
do
    sbatch -c 2 --gres=gpu:1 -o output_files/baselines/hinton_baseline_seed_$i.out -J hinton$i scripts/run_hinton_seed.sh $i
done

# for i in {0..4}
# do
#     sbatch -c 2 --gres=gpu:1 -w lambda1 -o output_files/baselines/hinton_baseline_seed_$i.out -J hinton$i scripts/run_hinton_seed.sh $i
#     sbatch -c 2 --gres=gpu:1 -w lambda1 -o output_files/baselines/no_teacher_baseline_seed_$i.out -J no_teacher$i scripts/run_noteacher_seed.sh $i
# done

# for i in {0..9}
# do
#     scancel 17354$i
# done