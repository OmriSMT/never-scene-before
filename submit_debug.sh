#!/bin/bash
#SBATCH --job-name=scene_debug
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scene_debug_%j.out
#SBATCH --error=logs/scene_debug_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate never_scene

echo "Node: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

bash run_squad_debug.sh

