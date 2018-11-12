#!/bin/bash
#SBATCH --job-name=SuperSAM
#SBATCH --output=Output-%A.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=samben@uni-hildesheim.de

# Never forget that! Strange happenings ensue otherwise.
#SBATCH --export=NONE
#SBATCH --partition=STUD
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

set -e

## here comes the program, you want to start :

echo "Training starts ..."

cd /home/samben/ismll_work/SRP/goprime/

/home/samben/anaconda3/envs/ismll/bin/python -c "import os; os.system('nvidia-smi')"

srun /home/samben/anaconda3/envs/ismll/bin/python -c "import tensorflow as tf; print(tf.contrib.eager.num_gpus())"

#srun /home/samben/anaconda3/envs/ismll/bin/python goprime.py -m selfplay

srun /home/samben/anaconda3/envs/ismll/bin/python goprime.py -m replay_traindist -b 0,250 -d ../dataset/
