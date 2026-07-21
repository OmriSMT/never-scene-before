#!/bin/bash

NUM_PERT=1
NUM_PERM=0
NUM_RETV=1

WEIGHT_PERT=1.0
WEIGHT_PERM=0.0
WEIGHT_RETV=1.0

SEED=42
EPOCHS=3
DATASET=squad
MODEL_NAME=csarron/roberta-base-squad-v1

OUTPUT_DIR=./checkpoints/${DATASET}/ner_retr_epochs${EPOCHS}_seed${SEED}_pert${NUM_PERT}_perm${NUM_PERM}_retr${NUM_RETV}

mkdir -p ${OUTPUT_DIR}

echo "SEED=${SEED}"
echo "NUM_PERT=${NUM_PERT}"
echo "NUM_PERM=${NUM_PERM}"
echo "NUM_RETV=${NUM_RETV}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

accelerate launch ../../train.py \
  --model_name_or_path ${MODEL_NAME} \
  --per_device_train_batch_size 32 \
  --num_train_epochs ${EPOCHS} \
  --seed ${SEED} \
  --learning_rate 2e-5 \
  --custom_warmup_steps 100 \
  --weight_decay 0.01 \
  --dataset_name ${DATASET} \
  --pad_to_max_length \
  --mask_strategy "ner" \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --num_perturbation_examples_per_batch ${NUM_PERT} \
  --num_permutation_examples_per_batch ${NUM_PERM} \
  --num_retrieval ${NUM_RETV} \
  --weight_perturb ${WEIGHT_PERT} \
  --weight_permute ${WEIGHT_PERM} \
  --weight_retrieval ${WEIGHT_RETV} \
  --retrieval_data_path ../../data/train-w-ret.csv \
  --retrieval_context_title_path ../../data/train_documents.csv \
  --remove_no_answer \
  --use_paraphrase_detector \
  --output_dir ${OUTPUT_DIR}