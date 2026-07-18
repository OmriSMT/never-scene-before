#!/usr/bin/env python
# coding=utf-8
"""Standalone evaluation script for a saved BoolQ classifier checkpoint.

Mirrors eval_checkpoint.py (the SQuAD evaluator), with
AutoModelForQuestionAnswering swapped for AutoModelForSequenceClassification.
Evaluation is on BoolQ3L -- the three-label (NO / YES / NO ANSWER) extension of
BoolQ (see boolq3l.py) -- so the IDK class the SCENE model is trained to predict
is actually scored, rather than the answerable-only google/boolq validation set.

Usage:
    accelerate launch eval_boolq.py \
        --model_name_or_path ./checkpoints/boolq/roberta_base_epochs2_seed42 \
        --split dev \
        --pad_to_max_length \
        --per_device_eval_batch_size 16

Reports overall accuracy and macro-F1 plus per-label precision / recall / F1 and
the gold / predicted label distributions, so IDK detection is visible.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

import datasets
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

from labels_boolq import ID2LABEL, NUM_LABELS
from boolq3l import load_boolq3l, load_boolq3l_all, SPLIT_FILES

logger = get_logger(__name__)


def save_prefixed_metrics(results, output_dir, file_name="all_results.json", metric_key_prefix="eval"):
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)


def per_label_metrics(preds, refs):
    """Precision / recall / F1 / support for each of the three labels."""
    metrics = {}
    macro_f1 = []
    for label_id in sorted(ID2LABEL):
        tp = int(np.sum((preds == label_id) & (refs == label_id)))
        fp = int(np.sum((preds == label_id) & (refs != label_id)))
        fn = int(np.sum((preds != label_id) & (refs == label_id)))
        support = int(np.sum(refs == label_id))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[ID2LABEL[label_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_f1.append(f1)
    return metrics, float(np.mean(macro_f1))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved BoolQ classifier checkpoint on BoolQ3L")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to saved checkpoint or HuggingFace model identifier.")
    parser.add_argument("--split", type=str, default="dev", choices=[*SPLIT_FILES, "all"],
                        help="Which BoolQ3L split to evaluate: dev | train | all "
                             "(default: dev). 'all' concatenates every split.")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Truncate the eval set (for quick testing).")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write results. Defaults to "
                             "<model_name_or_path>/eval_results/boolq3l.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator()

    output_dir = args.output_dir or os.path.join(args.model_name_or_path, "eval_results", "boolq3l")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logger.info(f"Loading BoolQ classifier {args.model_name_or_path}...")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if config.num_labels != NUM_LABELS:
        logger.warning(
            f"Checkpoint has {config.num_labels} labels but BoolQ3L needs {NUM_LABELS} "
            f"(NO / YES / NO ANSWER); the IDK class may not be predictable."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # --- load BoolQ3L (3-label) eval set ---
    eval_examples = load_boolq3l_all() if args.split == "all" else load_boolq3l(args.split)
    if args.max_eval_samples is not None:
        eval_examples = eval_examples.select(range(args.max_eval_samples))

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def _prepare(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["passage"],
            truncation="only_second",
            max_length=max_seq_length,
            padding="max_length" if args.pad_to_max_length else False,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            _prepare,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_examples.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Tokenizing BoolQ3L eval split",
        )

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    logger.info("***** Running Evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    # --- eval loop (same shape as train_boolq_base.py) ---
    model.eval()
    all_preds, all_refs = [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        preds, refs = accelerator.gather_for_metrics((preds, batch["labels"]))
        all_preds.append(preds.cpu().numpy())
        all_refs.append(refs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_refs = np.concatenate(all_refs)

    accuracy = float(np.mean(all_preds == all_refs))
    label_metrics, macro_f1 = per_label_metrics(all_preds, all_refs)

    def _dist(arr):
        return {ID2LABEL[i]: int(np.sum(arr == i)) for i in sorted(ID2LABEL)}

    results = {
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "num_examples": int(len(all_refs)),
        "per_label": label_metrics,
        "gold_distribution": _dist(all_refs),
        "prediction_distribution": _dist(all_preds),
    }

    if accelerator.is_main_process:
        logger.info(json.dumps(results, indent=4))

        predictions = [
            {"id": eid, "prediction": ID2LABEL[int(p)], "gold": ID2LABEL[int(r)]}
            for eid, p, r in zip(eval_examples["id"], all_preds, all_refs)
        ]
        with open(os.path.join(output_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=4)

        save_prefixed_metrics(dict(results), output_dir)


if __name__ == "__main__":
    main()