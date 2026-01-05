#!/bin/bash
cd /home/huaxi/scu_robotics/gaitnet_test

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaitnet

# Set Python path and library path
export PYTHONPATH=$PYTHONPATH:/home/huaxi/scu_robotics/gaitnet_test/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run viewer
cd build
./viewer/viewer ../data/env.xml