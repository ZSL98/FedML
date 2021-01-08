#!/bin/bash

#SBATCH --job-name=FedML_test
#SBATCH --partition=cpu
#SBATCH --mail-type=end
#SBATCH --mail-user=zslzsl1998@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --nodelist=cas[278-280]

#module load miniconda3
#source activate FedML

sh run_fedavg_distributed_pytorch.sh 10 10 1 4 lr hetero 200 20 10 0.03 mnist "./../../../data/mnist" 0

#python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'
