#!/bin/bash

#SBATCH --job-name=FedML
#SBATCH --partition=small
#SBATCH -n 6
#SBATCH --mail-type=end
#SBATCH --mail-user=zslzsl1998@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#module load miniconda3
#source activate FedML

sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0
#python train.py

#python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'
