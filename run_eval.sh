#!/bin/bash
#
# Evaluate a saved checkpoint on a chosen dataset.
#
# Usage:
#   bash run_eval.sh <checkpoint_dir> [dataset] [ace_whqa_split]
#
#   dataset:         squad1 | squad2 | ace-whqa   (default: squad2)
#   ace_whqa_split:  all | has-answer | competitive | non-competitive
#                    (only used when dataset is ace-whqa; default: all)
#
# Examples:
#   bash run_eval.sh ./checkpoints/my_model                       # SQuAD v2
#   bash run_eval.sh ./checkpoints/my_model squad1                # SQuAD 1.1
#   bash run_eval.sh ./checkpoints/my_model ace-whqa competitive  # ACE-whQA slice

CHECKPOINT_DIR=${1:?"Usage: bash run_eval.sh <checkpoint_dir> [squad1|squad2|ace-whqa] [ace_whqa_split]"}
DATASET=${2:-squad2}
ACE_SPLIT=${3:-all}

# Map the friendly dataset name to eval_checkpoint.py flags.
EXTRA_ARGS=""
case "${DATASET}" in
  squad1|squad)
    DATASET_NAME="squad"
    EVAL_TAG="squad1"
    ;;
  squad2|squad_v2)
    DATASET_NAME="squad_v2"
    EXTRA_ARGS="--version_2_with_negative"
    EVAL_TAG="squad2"
    ;;
  ace-whqa|ace)
    DATASET_NAME="ace-whqa"
    EXTRA_ARGS="--version_2_with_negative --ace_whqa_split ${ACE_SPLIT}"
    EVAL_TAG="ace-whqa-${ACE_SPLIT}"
    ;;
  *)
    echo "Unknown dataset '${DATASET}'. Choose one of: squad1, squad2, ace-whqa." >&2
    exit 1
    ;;
esac

# Keep results from different datasets/splits side by side.
OUTPUT_DIR=${CHECKPOINT_DIR}/eval_results/${EVAL_TAG}
mkdir -p ${OUTPUT_DIR}

python eval_checkpoint.py \
  --model_name_or_path ${CHECKPOINT_DIR} \
  --dataset_name ${DATASET_NAME} \
  ${EXTRA_ARGS} \
  --pad_to_max_length \
  --max_seq_length 384 \
  --doc_stride 128 \
  --preprocessing_num_workers 1 \
  --per_device_eval_batch_size 16 \
  --output_dir ${OUTPUT_DIR}
