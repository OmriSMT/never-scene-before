#!/bin/bash

source ~/.bashrc
conda activate never_scene

NUM_PERT=1
NUM_PERM=0
NUM_RETV=0

WEIGHT_PERT=1.0
WEIGHT_PERM=0.0
WEIGHT_RETV=0.0

EPOCHS=1
DATASET=squad
MODEL_NAME=csarron/roberta-base-squad-v1

MODEL_SHORT=$(basename ${MODEL_NAME})
OUTPUT_DIR=./checkpoints/${DATASET}/${MODEL_SHORT}_pert_${NUM_PERT}w${WEIGHT_PERT}_perm_${NUM_PERM}w${WEIGHT_PERM}_retr_${NUM_RETV}w${WEIGHT_RETV}_epochs_${EPOCHS}

mkdir -p ${OUTPUT_DIR}

echo "=== Training ==="
accelerate launch train.py \
  --model_name_or_path ${MODEL_NAME} \
  --per_device_train_batch_size 8 \
  --num_train_epochs ${EPOCHS} \
  --learning_rate 2e-5 \
  --custom_warmup_steps 0 \
  --weight_decay 0.01 \
  --dataset_name ${DATASET} \
  --pad_to_max_length \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --num_perturbation_examples_per_batch ${NUM_PERT} \
  --num_permutation_examples_per_batch ${NUM_PERM} \
  --num_retrieval ${NUM_RETV} \
  --weight_perturb ${WEIGHT_PERT} \
  --weight_permute ${WEIGHT_PERM} \
  --weight_retrieval ${WEIGHT_RETV} \
  --remove_no_answer \
  --use_paraphrase_detector \
  --output_dir ${OUTPUT_DIR}

if [ $? -ne 0 ]; then
  echo "Training failed. Skipping evaluation."
  exit 1
fi

echo "=== Evaluation ==="
bash run_eval.sh ${OUTPUT_DIR}