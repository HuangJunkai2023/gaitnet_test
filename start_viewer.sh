#!/bin/bash
cd /home/huaxi/scu_robotics/gaitnet_test

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaitnet_new

# Set Python path
export PYTHONPATH=$PYTHONPATH:/home/huaxi/scu_robotics/gaitnet_test/python

# Run viewer
cd build
./viewer/viewer ../data/env.xml