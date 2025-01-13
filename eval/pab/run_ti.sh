#!/bin/bash

# Read arguments
arg_index=$1

# Print the arguments
echo "TI evaluation number: $arg_index"

# Configure visible cuda devices
export CUDA_VISIBLE_DEVICES=$(($arg_index%8))
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the evaluation
python experiments/tokenwise.py --index $arg_index
