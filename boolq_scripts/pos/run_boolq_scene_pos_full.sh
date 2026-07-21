#!/bin/bash

NUM_PERT=${NUM_PERT:-1}
NUM_PERM=${NUM_PERM:-1}
 
WEIGHT_PERT=${WEIGHT_PERT:-1.0}
WEIGHT_PERM=${WEIGHT_PERM:-1.0}
 
EPOCHS=10
DATASET_NAME=google/boolq
MODEL_NAME=EyalMaor/roberta-base-boolq-idk
 
MASK_STRATEGY=pos
SEED=${SEED:-42}

POS_TAGS=${POS_TAGS:-"NOUN PROPN ADJ NUM"}
POS_NAME=${POS_NAME:-"no_verb"}

OUTPUT_DIR=./checkpoints/boolq/pos_${POS_NAME}_pert${NUM_PERT}_perm${NUM_PERM}_epoch${EPOCHS}_seed${SEED}

echo "MASK_STRATEGY=${MASK_STRATEGY}"
echo "POS_NAME=${POS_NAME}"
echo "POS_TAGS=${POS_TAGS}"
echo "SEED=${SEED}"
echo "NUM_PERT=${NUM_PERT}"
echo "NUM_PERM=${NUM_PERM}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
 
mkdir -p ${OUTPUT_DIR}
 
accelerate launch ../../train_boolq_scene.py \
  --model_name_or_path ${MODEL_NAME} \
  --dataset_name ${DATASET_NAME} \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs ${EPOCHS} \
  --learning_rate 1e-5 \
  --custom_warmup_steps 100 \
  --seed ${SEED} \
  --weight_decay 0.01 \
  --pad_to_max_length \
  --mask_strategy ${MASK_STRATEGY} \
  --pos_tags ${POS_TAGS} \
  --max_seq_length 256 \
  --doc_stride 128 \
  --num_perturbation_examples_per_batch ${NUM_PERT} \
  --num_permutation_examples_per_batch ${NUM_PERM} \
  --weight_perturb ${WEIGHT_PERT} \
  --weight_permute ${WEIGHT_PERM} \
  --checkpointing_steps epoch \
  --output_dir ${OUTPUT_DIR}