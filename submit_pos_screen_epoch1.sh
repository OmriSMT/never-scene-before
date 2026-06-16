#!/bin/bash
#SBATCH --job-name=pos_screen1
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pos_screen_epoch1_%j.out
#SBATCH --error=logs/pos_screen_epoch1_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate never_scene

echo "Node: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "POS_NAME=${POS_NAME}"
echo "POS_TAGS=${POS_TAGS}"

nvidia-smi

bash run_squad_pos_screen_epoch1.sh
