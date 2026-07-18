#!/bin/bash
#SBATCH --job-name=scene_pos_full3
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pos_full_epoch3_%j.out
#SBATCH --error=logs/pos_full_epoch3_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate never_scene

echo "Node: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

bash run_squad_pos_full_epoch3.sh
