#!/bin/bash
#
#
# Usage: run_training.sh
# Description: This script runs the training of the model with different CLA
#

# List of 'which_data' values, 'resampled_sample' , 'sim_output', 'bench_image'
which_data_list=("resampled_sample" "sim_output" "bench_image")
# List of pretrained or not pretrained classifier
transfer_learn_list=(1 0)

# For each of the which_data values, call train.py --which_data {which_data}
for which_data in "${which_data_list[@]}"
do
    # For each of the transfer_learn values, call train.py --transfer_learn {transfer_learn}
    for transfer_learn in "${transfer_learn_list[@]}"
    do
        python train.py --which_data $which_data --transfer_learn $transfer_learn
    done
done
