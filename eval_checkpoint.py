#!/usr/bin/env python
# coding=utf-8
"""Standalone evaluation script for a saved QA checkpoint on SQuAD / SQuAD v2."""

import argparse
import json
import logging
import os
from functools import partial
from collections import defaultdict
import numpy as np
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

from ace_whqa import load_ace_whqa, load_ace_whqa_all, SPLIT_FILES
from dataloading import prepare_validation_features
from eval_utils import run_evaluation

# Friendly split names accepted by --ace_whqa_split (besides "all").
ACE_WHQA_SPLITS = list(SPLIT_FILES)

logger = get_logger(__name__)


def save_prefixed_metrics(results, output_dir, file_name="all_results.json", metric_key_prefix="eval"):
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved QA model checkpoint")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to saved checkpoint or HuggingFace model identifier.")
    parser.add_argument("--dataset_name", type=str, default="squad_v2",
                        help="HuggingFace dataset name (default: squad_v2), or 'ace-whqa' to "
                             "evaluate on the local ACE-whQA corpus (see --ace_whqa_split).")
    parser.add_argument("--ace_whqa_split", type=str, default="all",
                        choices=["all", *ACE_WHQA_SPLITS],
                        help="Which ACE-whQA slice to evaluate when --dataset_name ace-whqa "
                             "(default: all = the three slices concatenated).")
    parser.add_argument("--version_2_with_negative", action="store_true",
                        help="Use SQuAD v2 metric (with unanswerable questions).")
    parser.add_argument("--mask_strategy", type=str, required=True,
                        choices=['random', 'loss', 'pos', 'ner'],
                        help=f"Masking strategy (choices: {', '.join(['random', 'loss', 'pos', 'ner'])})")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--no_answer_probability_threshold", type=float, default=0.5)
    parser.add_argument("--use_threshold", action="store_true")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Truncate validation set (for quick testing).")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

    model_name_or_path = args.model_name_or_path
    output_dir = os.path.join(model_name_or_path, "eval_results", args.dataset_name)

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

    if args.dataset_name == "ace-whqa":
        if not args.version_2_with_negative:
            logger.warning(
                "ACE-whQA contains unanswerable questions; you probably want "
                "--version_2_with_negative for correct (SQuAD v2) scoring."
            )
        if args.ace_whqa_split == "all":
            ace_validation = load_ace_whqa_all()
        else:
            ace_validation = load_ace_whqa(args.ace_whqa_split)
        logger.info(f"Loaded ACE-whQA split '{args.ace_whqa_split}' ({len(ace_validation)} examples)")
        raw_datasets = datasets.DatasetDict({"validation": ace_validation})
    else:
        raw_datasets = load_dataset(args.dataset_name)

    merged_dict = defaultdict(list)
    checkpoints_parent_dir = os.path.join("squad_scripts", args.mask_strategy)

    for model_seed in [model_seed for model_seed in os.listdir(os.path.join(checkpoints_parent_dir, "checkpoints", "squad")) if model_name_or_path in model_seed]:
        logger.info(f"Evaluating model checkpoint: {model_seed}")
        model_seed_path = os.path.join(checkpoints_parent_dir, "checkpoints", "squad", model_seed)
        config = AutoConfig.from_pretrained(model_seed_path)
        tokenizer = AutoTokenizer.from_pretrained(model_seed_path, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(model_seed_path, config=config)

        column_names = raw_datasets["validation"].column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        pad_on_right = tokenizer.padding_side == "right"
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

        prepare_validation_features_fn = partial(
            prepare_validation_features,
            tokenizer=tokenizer,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            pad_on_right=pad_on_right,
            max_seq_length=max_seq_length,
            args=args,
        )

        eval_examples = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(args.max_eval_samples))

        with accelerator.main_process_first():
            eval_dataset = eval_examples.map(
                prepare_validation_features_fn,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        if args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
        eval_dataloader = DataLoader(
            eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

        metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        logger.info("***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        eval_metric = run_evaluation(
            model, eval_dataloader, eval_dataset, eval_examples,
            accelerator, metric, args, logger, answer_column_name, output_dir
        )
        for key, value in eval_metric.items():
            merged_dict[key].append(value)

        if accelerator.is_main_process:
            logger.info(json.dumps(eval_metric, indent=4))


    if accelerator.is_main_process:
        # Average metrics across seeds
        logger.info(f"Averaged metrics across all evaluated checkpoints for {args.model_name_or_path}:")
        averaged_metrics = {key: (np.mean(values), np.std(values)) for key, values in merged_dict.items()}
        logger.info(json.dumps(averaged_metrics, indent=4))
        # save_prefixed_metrics(averaged_metrics, output_dir)


if __name__ == "__main__":
    main()
