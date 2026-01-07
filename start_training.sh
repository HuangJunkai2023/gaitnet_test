#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaitnet

# 设置 Python 路径和库路径
export PYTHONPATH=$PYTHONPATH:/home/huaxi/scu_robotics/gaitnet_test/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 进入 python 目录（环境文件使用相对路径 ../data/）
cd /home/huaxi/scu_robotics/gaitnet_test/python

# 运行训练脚本
python ray_train.py --config ppo --config-file ray_config.py --name emg_reward_training --env ../data/env.xml "$@" #--checkpoint ./ray_results/emg_reward_training/
