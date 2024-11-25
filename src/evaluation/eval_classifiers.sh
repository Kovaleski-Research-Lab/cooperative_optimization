#!/bin/bash


# Evaluate the classifiers
#checkpoint_path = '/devleop/results/classifier_baseline_bench_resampled_sample/version_1/'

python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_resampled_sample/version_2/'
python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_resampled_sample/version_3/'
python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_resampled_sample/version_4/'

python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_sim_output/version_2/'
python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_sim_output/version_3/'
python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_sim_output/version_4/'

python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_bench_image/version_2/'
python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_bench_image/version_3/'
python /develop/code/cooperative_optimization/src/evaluation/evaluation.py --checkpoint_path '/develop/results/classifier_baseline_bench_bench_image/version_4/'

