import math

import torch
from transformers import get_scheduler
from accelerate import Accelerator
import logging
import argparse


def create_optimizer_and_scheduler_boolq(model, args, train_dataloader):
    """Same weight-decay grouping as optimization.py's create_optimizers_and_scheduler,
    minus the generator optimizer -- BART and the QQP classifier stay frozen here,
    they're only used to synthesize perturbed examples, not trained themselves."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    return optimizer, lr_scheduler, overrode_max_train_steps


def calculate_and_backward_perturb_loss_boolq(model, perturbed_batch, pseudo_labels, keep_mask, accelerator, args, logger):
    """
    Cross-entropy on the self-labeled perturbed batch, masked to only the
    examples that passed the SCENE-style filter, weighted by args.weight_perturb.
    Mirrors optimization.calculate_and_backward_perturb_loss, but for a single
    scalar class label per example instead of start/end span logits.
    """
    if keep_mask.sum() == 0:
        logger.info("perturb_boolq loss: no examples passed the filter this step, skipping")
        return

    outputs = model(input_ids=perturbed_batch["input_ids"], attention_mask=perturbed_batch["attention_mask"])
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_example_loss = loss_fct(outputs.logits, pseudo_labels) * keep_mask
    p_loss = per_example_loss.sum() / keep_mask.sum()
    p_loss = p_loss * args.weight_perturb

    accelerator.backward(p_loss)
    logger.info(f"perturbed loss: {p_loss.detach().float()} (kept {int(keep_mask.sum().item())} examples)")


def calculate_and_backward_permute_loss(model, batch, tokenizer, accelerator, args, max_seq_length, logger):
    """
    BoolQ counterpart of optimization.py's calculate_and_backward_permute_loss.
    BoolQ has no "no answer" class to fall back on, so mismatched (question, permuted_passage) pairs
    are instead self-labeled with the model's own current prediction on that new pairing 
    which is the same self-labeling idea as the counterfactual at the og purtu 
    Only entries whose passage actually changed (i.e. the permutation didn't map them back to themselves) contribute to the loss.
    """
    device = batch["input_ids"].device
    batch_size = batch["input_ids"].shape[0]
 
    ids = torch.arange(batch_size)
    perm_ids = torch.randperm(batch_size)
    changed_mask = (ids != perm_ids).float().to(device)
 
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    original = tokenizer.batch_decode(batch["input_ids"])
    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip() for x in original]
    passages = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip() for x in original]
    permuted_passages = [passages[i] for i in perm_ids.tolist()]
 
    try:
        tokenized = tokenizer(
            questions,
            permuted_passages,
            truncation="only_second",
            max_length=max_seq_length,
            stride=args.doc_stride,
            padding="max_length" if args.pad_to_max_length else True,
            return_tensors="pt",
        ).to(device)
    except Exception:
        logger.info("Failed permutation batch; skipping this step's permute loss")
        return
 
    outputs = model(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"])
    with torch.no_grad():
        pseudo_labels = outputs.logits.argmax(dim=-1)  # self-labeled, same trick as the SCENE branch
 
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_example_loss = loss_fct(outputs.logits, pseudo_labels) * changed_mask
    denom = changed_mask.sum()
    loss = per_example_loss.sum() / denom if denom > 0 else 0.0 * per_example_loss.sum()
    loss = loss * args.weight_permute / max(args.num_permutation_examples_per_batch, 1)
 
    accelerator.backward(loss)
    logger.info(f"permute loss: {loss.detach().float()} ({int(changed_mask.sum().item())} changed pairs)")
