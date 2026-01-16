#!/bin/bash
cd /home/hx/code/gaitnet_test

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaitnet_new

# Set Python path and library path
export PYTHONPATH=$PYTHONPATH:/home/hx/code/gaitnet_test/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run viewer
cd build
# ./viewer/viewer ../data/env.xml

./viewer/viewer ../emg_trained_model/retrained_9000

# ./viewer/viewer ../data/trained_nn/merge_no_mesh_lbs
