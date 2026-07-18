#!/bin/bash

SEED=42
EPOCHS=2

DATASET_NAME=google/boolq      # passed to --dataset_name / load_dataset(...)
DATASET_LABEL=boolq            # short label, used only for the output dir name
MODEL_NAME=roberta-base

MAX_SEQ_LENGTH=320
TRAIN_BATCH_SIZE=12
EVAL_BATCH_SIZE=16
LEARNING_RATE=3e-5
WEIGHT_DECAY=0.01

OUTPUT_DIR=./checkpoints/${DATASET_LABEL}/roberta_base_epochs${EPOCHS}_seed${SEED}

mkdir -p ${OUTPUT_DIR}

echo "SEED=${SEED}"
echo "EPOCHS=${EPOCHS}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

accelerate launch train_boolq_accelerate.py \
  --dataset_name ${DATASET_NAME} \
  --model_name_or_path ${MODEL_NAME} \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --num_train_epochs ${EPOCHS} \
  --seed ${SEED} \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --pad_to_max_length \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --doc_stride 128 \
  --output_dir ${OUTPUT_DIR}
