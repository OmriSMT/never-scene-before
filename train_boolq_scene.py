#!/usr/bin/env python
# coding=utf-8
"""
Fine-tune a BoolQ classifier (e.g. shahrukhx01/roberta-base-boolq) with
SCENE-style self-labeled counterfactual augmentation, adapted from
never-scene-before's train.py.

See models_boolq.py / dataloading_boolq.py / perturb_boolq.py /
optimization_boolq.py / mask_strategies_boolq.py for the pieces this
composes, and the accompanying chat message for the QA -> classification
mapping this implements.

Usage:
    python train_boolq_scene.py \
        --model_name_or_path shahrukhx01/roberta-base-boolq \
        --dataset_name boolq \
        --output_dir ./checkpoints/boolq-scene \
        --mask_strategy pos \
        --num_perturbation_examples_per_batch 1 \
        --weight_perturb 1.0 --weight_permute 1.0 \
        --per_device_train_batch_size 8 \
        --num_train_epochs 3
"""
import json
import logging
import math
import os

import datasets
import evaluate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm


from models_boolq import load_models_boolq
from dataloading_boolq import load_boolq_datasets, preprocess_boolq, build_dataloaders_boolq
from mask_strategies_boolq import (
    RandomMaskStrategy, POSMaskStrategy, NERMaskStrategy, ClassificationLossMaskStrategy,
)
from perturb_boolq import evaluate_and_filter_perturbations_boolq
from args import parse_args
from optimization_boolq import create_optimizer_and_scheduler_boolq, calculate_and_backward_perturb_loss_boolq, calculate_and_backward_permute_loss

logger = get_logger(__name__)

MASK_STRATEGIES = {
    "random": RandomMaskStrategy,
    "loss": ClassificationLossMaskStrategy,
    "ner": NERMaskStrategy,
    "pos": POSMaskStrategy,
}


def save_prefixed_metrics(results, output_dir, file_name="all_results.json", metric_key_prefix="eval"):
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)



def run_eval(model, eval_dataloader, accelerator):
    metric = evaluate.combine(["accuracy", "f1"])
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        preds, labels = accelerator.gather_for_metrics((preds, batch["labels"]))
        metric.add_batch(predictions=preds, references=labels)
    return metric.compute()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # -------------------------------------------------------------------------
    # Data + models
    # -------------------------------------------------------------------------
    raw_datasets = load_boolq_datasets(args)
    model, tokenizer, generator, generator_tokenizer, paraphrase_classifier, paraphrase_tokenizer = \
        load_models_boolq(args)

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    train_dataset, eval_dataset = preprocess_boolq(args, raw_datasets, tokenizer, accelerator, max_seq_length)
    train_dataloader, eval_dataloader = build_dataloaders_boolq(args, train_dataset, eval_dataset, tokenizer, accelerator)

    # -------------------------------------------------------------------------
    # Mask strategy
    # -------------------------------------------------------------------------
    # validate the mask strategy
    strategy_cls = MASK_STRATEGIES.get(args.mask_strategy, None)
    if strategy_cls is None:
        raise Exception(f"Mask strategy {args.mask_strategy} is not supported.")
    elif args.mask_strategy == "loss":
        mask_strategy = strategy_cls(model, tokenizer, max_seq_length)
        logger.info(f"Using {args.mask_strategy} mask strategy for perturbation.")
    elif args.mask_strategy == "ner":
        mask_strategy = strategy_cls(target_ents=args.ner_labels)
        logger.info(f"Using {args.mask_strategy} mask strategy for perturbation.")
        logger.info(f"NER labels used for masking: {args.ner_labels}")
    elif args.mask_strategy == "pos":
        mask_strategy = strategy_cls(target_pos=args.pos_tags)
        logger.info(f"Using {args.mask_strategy} mask strategy for perturbation.")
        logger.info(f"POS labels used for masking: {args.pos_tags}")
    else:
        mask_strategy = strategy_cls()
        logger.info(f"Using {args.mask_strategy} mask strategy for perturbation.")

    # -------------------------------------------------------------------------
    # Optimizer / scheduler / accelerate prepare
    # -------------------------------------------------------------------------
    optimizer, lr_scheduler, overrode_max_train_steps = create_optimizer_and_scheduler_boolq(model, args, train_dataloader)

    model, generator, paraphrase_classifier, optimizer, train_dataloader, eval_dataloader, lr_scheduler = \
        accelerator.prepare(model, generator, paraphrase_classifier, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    generator.eval()
    paraphrase_classifier.eval()
    model.train()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            no_pert = (step <= args.custom_warmup_steps and epoch == 0) or args.num_perturbation_examples_per_batch == 0

            if not no_pert:
                model.eval()
                perturbed_batch, pseudo_labels, keep_mask, info = evaluate_and_filter_perturbations_boolq(
                    batch=batch, model=model, tokenizer=tokenizer,
                    generator_tokenizer=generator_tokenizer, generator=generator,
                    paraphrase_tokenizer=paraphrase_tokenizer, paraphrase_classifier=paraphrase_classifier,
                    args=args, max_seq_length=max_seq_length, mask_strategy=mask_strategy,
                    logger=logger, num_processes=accelerator.num_processes,
                )
                model.train()

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                logger.info(f"model loss: {loss.detach().float()}")

                if not no_pert:
                    calculate_and_backward_perturb_loss_boolq(
                        model, perturbed_batch, pseudo_labels, keep_mask, accelerator, args, logger,
                    )

                    if args.num_permutation_examples_per_batch > 0 and args.weight_permute > 0:
                        for _ in range(args.num_permutation_examples_per_batch):
                            calculate_and_backward_permute_loss(
                                model, batch, tokenizer, accelerator, args, max_seq_length, logger,
                            )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"step_{completed_steps}"))

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

    # -------------------------------------------------------------------------
    # Final validation + save
    # -------------------------------------------------------------------------
    eval_metric = run_eval(model, eval_dataloader, accelerator)
    if accelerator.is_main_process:
        logger.info(json.dumps(eval_metric, indent=4))

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        save_prefixed_metrics(eval_metric, args.output_dir)


if __name__ == "__main__":
    main()
