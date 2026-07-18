#!/bin/bash

NUM_PERT=${NUM_PERT:-1}
NUM_PERM=${NUM_PERM:-1}
NUM_RETV=${NUM_RETV:-1}

WEIGHT_PERT=${WEIGHT_PERT:-1.0}
WEIGHT_PERM=${WEIGHT_PERM:-1.0}
WEIGHT_RETV=${WEIGHT_RETV:-1.0}

EPOCHS=3
DATASET=squad
MODEL_NAME=csarron/roberta-base-squad-v1

MASK_STRATEGY=pos
POS_TAGS=${POS_TAGS:-"NOUN PROPN ADJ NUM"}
POS_NAME=${POS_NAME:-"no_verb"}
SEED=${SEED:-42}
CONFIG_NAME=${CONFIG_NAME:-"full_pipeline"}

OUTPUT_DIR=./checkpoints/${DATASET}/pos_${POS_NAME}_${CONFIG_NAME}_epoch3_seed${SEED}

echo "POS_NAME=${POS_NAME}"
echo "POS_TAGS=${POS_TAGS}"
echo "SEED=${SEED}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "NUM_PERT=${NUM_PERT}"
echo "NUM_PERM=${NUM_PERM}"
echo "NUM_RETV=${NUM_RETV}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

mkdir -p ${OUTPUT_DIR}

accelerate launch ../../train.py \
  --model_name_or_path ${MODEL_NAME} \
  --per_device_train_batch_size 32 \
  --num_train_epochs ${EPOCHS} \
  --learning_rate 2e-5 \
  --custom_warmup_steps 100 \
  --seed ${SEED} \
  --weight_decay 0.01 \
  --dataset_name ${DATASET} \
  --pad_to_max_length \
  --mask_strategy ${MASK_STRATEGY} \
  --pos_tags ${POS_TAGS} \
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
