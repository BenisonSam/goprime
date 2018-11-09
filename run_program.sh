#!/bin/bash
#SBATCH --job-name=BSeAlfHPlFay-2
#SBATCH --output=Output-%A.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=samben@uni-hildesheim.de

# Never forget that! Strange happenings ensue otherwise.
#SBATCH --export=NONE
#SBATCH --partition=stud
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

set -e

## here comes the program, you want to start :

echo "Training starts ..."

cd /home/samben/ismll_work/SRP/goprime/

conda activate ismll

python3 -c 'print("\n~ Testing NVIDIA ~\n")print(os.system("nvidia-smi"))
print("\n~ Testing TensorFlow ~\n")
import tensorflow as tf
print(tf.contrib.eager.num_gpus())'

python3 goprime.py 9 selfplay
