#!/bin/bash

NUM_PERT=1
NUM_PERM=0
NUM_RETV=0
MASK_STRATEGY=${MASK_STRATEGY:-random}
POS_TAGS=${POS_TAGS:-"NOUN PROPN VERB ADJ NUM"}
WEIGHT_PERT=1.0
WEIGHT_PERM=0.0
WEIGHT_RETV=0.0
EPOCHS=1
DATASET=squad
MODEL_NAME=csarron/roberta-base-squad-v1

OUTPUT_DIR=./checkpoints/${DATASET}/debug_${MASK_STRATEGY}_masking
mkdir -p ${OUTPUT_DIR}

accelerate launch train.py \
  --model_name_or_path ${MODEL_NAME} \
  --per_device_train_batch_size 2 \
  --num_train_epochs ${EPOCHS} \
  --max_train_steps 10 \
  --learning_rate 2e-5 \
  --custom_warmup_steps 0 \
  --weight_decay 0.01 \
  --dataset_name ${DATASET} \
  --pad_to_max_length \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --num_perturbation_examples_per_batch ${NUM_PERT} \
  --mask_strategy ${MASK_STRATEGY} \
  --pos_tags ${POS_TAGS} \
  --num_permutation_examples_per_batch ${NUM_PERM} \
  --num_retrieval ${NUM_RETV} \
  --weight_perturb ${WEIGHT_PERT} \
  --weight_permute ${WEIGHT_PERM} \
  --weight_retrieval ${WEIGHT_RETV} \
  --remove_no_answer \
  --use_paraphrase_detector \
  --output_dir ${OUTPUT_DIR}
  