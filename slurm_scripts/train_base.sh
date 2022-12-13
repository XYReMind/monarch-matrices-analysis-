#!/bin/bash

#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=monarch_base

NET=$1

module purge

singularity exec --nv \
	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; wandb offline; python train_cifar10.py \
	--lr 1e-4 \
	--opt adam \
	--patch 4 \
	--net $NET \
	--mixup \
	--wandbentity aaj458"


