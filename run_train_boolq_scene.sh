#!/bin/bash
# Usage: bash run_train_boolq_scene.sh [model_name_or_path] [mask_strategy]


MODEL_NAME_OR_PATH=${1:-"shahrukhx01/roberta-base-boolq"}
MASK_STRATEGY=${2:-"pos"}

python train_boolq_scene.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_name boolq \
    --max_seq_length 256 \
    --pad_to_max_length \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --mask_strategy "$MASK_STRATEGY" \
    --num_perturbation_examples_per_batch 1 \
    --weight_perturb 1.0 \
    --weight_permute 1.0 \
    --custom_warmup_steps 50 \
    --checkpointing_steps epoch \
    --output_dir "./checkpoints/boolq-scene"
