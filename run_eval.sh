#!/bin/bash

CHECKPOINT_DIR=${1:?"Usage: bash run_eval.sh <checkpoint_dir>"}
OUTPUT_DIR=${CHECKPOINT_DIR}/eval_results

mkdir -p ${OUTPUT_DIR}

python eval_checkpoint.py \
  --model_name_or_path ${CHECKPOINT_DIR} \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --pad_to_max_length \
  --max_seq_length 384 \
  --doc_stride 128 \
  --preprocessing_num_workers 1 \
  --per_device_eval_batch_size 16 \
  --output_dir ${OUTPUT_DIR}
