#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaitnet_new

# 设置 Python 路径和库路径
export PYTHONPATH=$PYTHONPATH:/home/huang/scu_robotics/gaitnet_test/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 设置使用GPU 0
export CUDA_VISIBLE_DEVICES=0

# 进入 python 目录（环境文件使用相对路径 ../data/）
cd /home/huang/scu_robotics/gaitnet_test/python

# 运行训练脚本
python ray_train.py --config ppo_small_pc --config-file ray_config.py --name emg_reward_training --env ../data/env.xml "$@" #--checkpoint ./ray_results/emg_reward_training/
