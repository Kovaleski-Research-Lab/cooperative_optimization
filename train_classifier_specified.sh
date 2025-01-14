#!/bin/bash

python train_classifier.py --which_data resampled_sample --transfer_learn 0 --freeze_backbone 0 --crop_normalize 1
python train_classifier.py --which_data resampled_sample --transfer_learn 1 --freeze_backbone 0 --crop_normalize 1
python train_classifier.py --which_data resampled_sample --transfer_learn 1 --freeze_backbone 1 --crop_normalize 1
python train_classifier.py --which_data sim_output --transfer_learn 0 --freeze_backbone 0 --crop_normalize 1
python train_classifier.py --which_data sim_output --transfer_learn 1 --freeze_backbone 0 --crop_normalize 1
python train_classifier.py --which_data sim_output --transfer_learn 1 --freeze_backbone 1 --crop_normalize 1
python train_classifier.py --which_data bench_image --transfer_learn 0 --freeze_backbone 0 --crop_normalize 1
python train_classifier.py --which_data bench_image --transfer_learn 1 --freeze_backbone 0 --crop_normalize 1
python train_classifier.py --which_data bench_image --transfer_learn 1 --freeze_backbone 1 --crop_normalize 1

python train_classifier.py --which_data resampled_sample --transfer_learn 0 --freeze_backbone 0 --crop_normalize 0
python train_classifier.py --which_data resampled_sample --transfer_learn 1 --freeze_backbone 0 --crop_normalize 0
python train_classifier.py --which_data resampled_sample --transfer_learn 1 --freeze_backbone 1 --crop_normalize 0
python train_classifier.py --which_data sim_output --transfer_learn 0 --freeze_backbone 0 --crop_normalize 0
python train_classifier.py --which_data sim_output --transfer_learn 1 --freeze_backbone 0 --crop_normalize 0
python train_classifier.py --which_data sim_output --transfer_learn 1 --freeze_backbone 1 --crop_normalize 0
python train_classifier.py --which_data bench_image --transfer_learn 0 --freeze_backbone 0 --crop_normalize 0
python train_classifier.py --which_data bench_image --transfer_learn 1 --freeze_backbone 0 --crop_normalize 0
python train_classifier.py --which_data bench_image --transfer_learn 1 --freeze_backbone 1 --crop_normalize 0

