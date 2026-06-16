#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import json
import logging
import math
from multiprocessing import context
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import pdb
import pandas as pd
import copy

import datasets
import evaluate
import transformers
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository


from args import parse_args, save_prefixed_metrics
from models import load_models
from dataloading import (
    load_raw_datasets,
    load_retrieval_dataset,
    preprocess_datasets,
    preprocess_retrieval_dataset,
    build_dataloaders,
)
from perturb import evaluate_and_filter_perturbations
from optimization import (create_optimizers_and_scheduler, calculate_and_backward_retrieval_loss,
                          calculate_and_backward_permute_loss, calculate_and_backward_perturb_loss)
from eval_utils import run_evaluation
from mask_strategies import RandomMaskStrategy, LossMaskStrategy # TODO: import more here when we have them implemented


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")
logger = get_logger(__name__)


# The possible mask strategies to choose from the argparser
MASK_STRATEGIES = {
    "random": RandomMaskStrategy,
    "loss": LossMaskStrategy,
}


def main():
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_qa_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    # # Handle the repository creation
    # if accelerator.is_main_process:
        # if args.push_to_hub:
        #     if args.hub_model_id is None:
        #         repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
        #     else:
        #         repo_name = args.hub_model_id
        #     repo = Repository(args.output_dir, clone_from=repo_name)
        #
        #     with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
        #         if "step_*" not in gitignore:
        #             gitignore.write("step_*\n")
        #         if "epoch_*" not in gitignore:
        #             gitignore.write("epoch_*\n")
        # elif args.output_dir is not None:
        #     os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    raw_datasets = load_raw_datasets(args)
    column_names = raw_datasets["train"].column_names
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    retrieval_dataset = None
    if args.num_retrieval > 0:
        retrieval_dataset = load_retrieval_dataset(args)

    # -------------------------------------------------------------------------
    # Model / tokenizer loading
    # -------------------------------------------------------------------------
    model, tokenizer, generator, generator_tokenizer, paraphrase_classifier, paraphrase_tokenizer = load_models(args)

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # -------------------------------------------------------------------------
    # Dataset preprocessing
    # -------------------------------------------------------------------------
    (
        train_dataset,
        validation_examples, validation_dataset,
        test_examples, test_dataset,
        column_names, answer_column_name,
    ) = preprocess_datasets(args, raw_datasets, tokenizer, accelerator, max_seq_length, pad_on_right)

    if args.num_retrieval > 0:
        retrieval_dataset = preprocess_retrieval_dataset(
            args, retrieval_dataset, tokenizer, accelerator, max_seq_length, pad_on_right, column_names
        )

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------
    train_dataloader, validation_dataloader, test_dataloader, retrieval_dataloader = build_dataloaders(
        args, train_dataset, validation_dataset, test_dataset, tokenizer, accelerator,
        retrieval_dataset=retrieval_dataset,
    )

    # -------------------------------------------------------------------------
    # Optimizer and learning rate scheduler
    # -------------------------------------------------------------------------
    optimizer, optim_gen, lr_scheduler = create_optimizers_and_scheduler(model, generator, args, train_dataloader)
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )
    generator, optim_gen = accelerator.prepare(generator, optim_gen)
    paraphrase_classifier = accelerator.prepare(paraphrase_classifier)


    if args.num_retrieval > 0:
        retrieval_dataloader = accelerator.prepare(
            retrieval_dataloader
        )

    overrode_max_train_steps = False
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("qa_no_trainer", experiment_config)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
    #         accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
    #         accelerator.load_state(args.resume_from_checkpoint)
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
    #         dirs.sort(key=os.path.getctime)
    #         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
    #     # Extract `epoch_{i}` or `step_{i}`
    #     training_difference = os.path.splitext(path)[0]

        # if "epoch" in training_difference:
        #     starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        #     resume_step = None
        # else:
        #     resume_step = int(training_difference.replace("step_", ""))
        #     starting_epoch = resume_step // len(train_dataloader)
        #     resume_step -= starting_epoch * len(train_dataloader)


    # validate the mask strategy
    strategy = MASK_STRATEGIES.get(args.mask_strategy, None)
    if strategy is None:
        raise Exception(f"Mask strategy {args.mask_strategy} is not supported.")
    elif args.mask_strategy == "loss":
        mask_strategy = strategy(model, tokenizer, max_seq_length)
        logger.info(f"Using {args.mask_strategy} mask strategy for perturbation.")
    else:
        mask_strategy = strategy()
        logger.info(f"Using {args.mask_strategy} mask strategy for perturbation.")
    
    generator.eval()
    paraphrase_classifier.eval()
    model.train()
    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.with_tracking:
            total_loss = 0
            total_gen_loss = 0
        
        if args.num_retrieval > 0:
            retrieval_dataloader_iterable = iter(retrieval_dataloader)

        for step, batch in enumerate(train_dataloader):
            # # We need to skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == starting_epoch:
            #     if resume_step is not None and step < resume_step:
            #         completed_steps += 1
            #         continue
            
            ###### compute model outputs for both original batch and perturb batch #####
            model.eval()
            # first run we calculate only regular loss
            no_pert_and_perm = (step <= args.custom_warmup_steps and epoch == 0)

            perturbation_info, mask = evaluate_and_filter_perturbations(
                batch=batch,
                model=model,
                tokenizer=tokenizer,
                generator_tokenizer=generator_tokenizer,
                generator=generator,
                paraphrase_tokenizer=paraphrase_tokenizer,
                paraphrase_classifier=paraphrase_classifier,
                args=args,
                max_seq_length=max_seq_length,
                pad_on_right=pad_on_right,
                num_processes=accelerator.num_processes,
                logger=logger,
                mask_strategy=mask_strategy,
            )
            
            model.train()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)
                logger.info(f"model loss: {loss.detach().float()}")

                if not no_pert_and_perm:
                    # Perturbed cases
                    if args.num_perturbation_examples_per_batch > 0 and args.weight_perturb > 0:
                        calculate_and_backward_perturb_loss(model, perturbation_info, accelerator, args, mask, logger)
                        
                    # adding permutation
                    if args.num_permutation_examples_per_batch > 0 and args.weight_permute > 0:
                        calculate_and_backward_permute_loss(model, batch, tokenizer, accelerator, args, max_seq_length,pad_on_right, logger)

                    # adding retrieval-based no answerable question
                    if args.num_retrieval > 0 and args.weight_retrieval > 0:
                        calculate_and_backward_retrieval_loss(model, retrieval_dataloader, retrieval_dataloader_iterable, accelerator, args, logger)

                # accumulate gradient and update the parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                # repo.push_to_hub(
                #     commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                # )

    # -------------------------------------------------------------------------
    # Validation Evaluation
    # -------------------------------------------------------------------------
    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    logger.info("***** Running Validation Evaluation *****")
    logger.info(f"  Num examples = {len(validation_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    val_metric = run_evaluation(model, validation_dataloader, validation_dataset, validation_examples, accelerator, metric, args, logger, answer_column_name, is_test=False)

    # -------------------------------------------------------------------------
    # Test Evaluation
    # -------------------------------------------------------------------------
    if args.do_predict:
        logger.info("***** Running Test Evaluation *****")
        logger.info(f"  Num examples = {len(test_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        test_metric = run_evaluation(model, test_dataloader, test_dataset, test_examples, accelerator, metric, args, logger, answer_column_name, is_test=True)

    if args.with_tracking:
        log = {
            "squad_v2" if args.version_2_with_negative else "squad": val_metric,
            "train_loss": total_loss.item() / len(train_dataloader),
            "generator_loss": total_gen_loss.item() / len(train_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        }
    if args.do_predict:
        log["squad_v2_predict" if args.version_2_with_negative else "squad_predict"] = test_metric

        accelerator.log(log, step=completed_steps)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            # if args.push_to_hub:
            #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            logger.info(json.dumps(val_metric, indent=4))
            save_prefixed_metrics(val_metric, args.output_dir)


if __name__ == "__main__":
    main()
