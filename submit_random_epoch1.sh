#!/bin/bash
#SBATCH --job-name=scene_rand_ep1
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/random_epoch1_%j.out
#SBATCH --error=logs/random_epoch1_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate never_scene

echo "Node: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

bash run_squad_random_epoch1.sh

