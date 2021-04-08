#!/usr/bin/env bash

#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 100 &
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --client_num_per_round 100 &
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --client_num_per_round 100 --t_1 0.6 --t_2 0 &

#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 50 &
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --client_num_per_round 50 &
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --client_num_per_round 50 --t_1 0.5 --t_2 0.2 &

python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_0_5.npy'&
python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_0_5.npy'&
python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_0_5.npy' --t_1 0.2 --t_2 0.1 &

python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_1_0.npy'&
python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_1_0.npy'&
python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_1_0.npy' --t_1 0.4 --t_2 0 &

python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_1_5.npy'&
python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_1_5.npy'&
python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --client_num_per_round 20 --imb_factor 0.1 --var_value './../../../proportions/emd_1000_1e-1_1_5.npy' --t_1 1 --t_2 0.4 &

#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 20 --var_value './../../../proportions/EMD_1000_1e-1_0_0.npy' --imb_factor 0.1&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 20 --var_value './../../../proportions/EMD_1000_2e-1_0_0.npy' --imb_factor 0.2&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --client_num_per_round 20 --var_value './../../../proportions/EMD_1000_5e-1_0_0.npy' --imb_factor 0.5&

