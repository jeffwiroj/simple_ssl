#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=pretrain
#SBATCH --mem=32G
#SBATCH --time=11:10:00
#SBATCH --cpus-per-task=4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python pretrain.py
"
