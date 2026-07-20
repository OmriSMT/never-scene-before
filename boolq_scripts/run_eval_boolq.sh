#!/bin/bash
#
# Evaluate a saved BoolQ classifier checkpoint on BoolQ3L (the three-label
# NO / YES / NO ANSWER extension of BoolQ).
#
# Usage:
#   bash run_eval_boolq.sh <checkpoint_dir> [split] <mask_strategy>
#
#   checkpoint_dir:  path to a saved checkpoint or a HuggingFace model id
#   split:           dev | train | all   (default: dev; 'all' concatenates
#                    every BoolQ3L split)
#   mask_strategy:   random | ner | pos | loss  (default: random)
#
# Examples:
#   bash run_eval_boolq.sh ./checkpoints/boolq/roberta_base_epochs2_seed42
#   bash run_eval_boolq.sh ./checkpoints/boolq/my_model all ner
#
# Results (all_results.json + predictions.json) are written to
#   eval_results/boolq3l/<checkpoint_dir>/

CHECKPOINT_DIR=${1:?"Usage: bash run_eval_boolq.sh <checkpoint_dir> [dev|train|all]"}
SPLIT=${2:-dev}
MASK_STRATEGY=${3:-random}

OUTPUT_DIR=eval_results/boolq3l/${CHECKPOINT_DIR}
mkdir -p ${OUTPUT_DIR}

python eval_boolq.py \
  --model_name_or_path ${CHECKPOINT_DIR} \
  --split ${SPLIT} \
  --mask_strategy ${} \
  --pad_to_max_length \
  --max_seq_length 384 \
  --preprocessing_num_workers 1 \
  --per_device_eval_batch_size 16 \
  --output_dir ${OUTPUT_DIR}
