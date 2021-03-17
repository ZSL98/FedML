#!/usr/bin/env bash

#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --vc_sample 32&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --prob_method 'y_max#1' --vc_sample 128 --client_optimizer adam_n --lr 0.0001&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --prob_method 'y_max#2' --vc_sample 128 --client_optimizer adam_n --lr 0.0001&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --prob_method 'y_max_mixed' --vc_sample 128 --client_optimizer adam_n --lr 0.0001&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --vc_sample 32&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --vc_sample 64&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --vc_sample 64&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 0 --vc_sample 128 --client_optimizer adam_n --lr 0.0001&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 2 --vc_sample 128 --client_optimizer adam_n --lr 0.0001

python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --prob_method 'y_max_mixed' --t_1 0 --t_2 1 --vc_sample 128 --client_optimizer adam_n --lr 0.0001&
python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --prob_method 'y_max#1' --vc_sample 128 --client_optimizer adam_n --lr 0.0001
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --prob_method 'y_max_mixed' --vc_sample 128 --client_optimizer adam_n --lr 0.0001&
#python ./fedml_experiments/standalone/fedavg/main_fedonline.py --random 1 --prob_method 'y_max_mixed' --vc_sample 128 --client_optimizer adam_n --lr 0.0001
