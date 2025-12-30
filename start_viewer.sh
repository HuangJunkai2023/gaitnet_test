#!/bin/bash
cd /home/huang/scu_robotics/BidirectionalGaitNet

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaitnet_new

# Set Python path and library path
export PYTHONPATH=$PYTHONPATH:/home/huang/scu_robotics/BidirectionalGaitNet/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run viewer
cd build
./viewer/viewer ../data/env.xml