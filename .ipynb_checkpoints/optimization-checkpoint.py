import math
import torch
from transformers import get_scheduler

from perturb import produce_no_answer_batch

def _is_boolq(args):
    return getattr(args, "dataset_name", "").lower() == "boolq"

def create_optimizers_and_scheduler(model, generator, args, train_dataloader):
    # Split weights in two groups, one with weight decay and the other not.
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

    optimizer_grouped_parameters_gen = [
        {
            "params": [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-4,
        },
        {
            "params": [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optim_gen = torch.optim.AdamW(optimizer_grouped_parameters_gen, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
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

    return optimizer, optim_gen, lr_scheduler

def calculate_and_backward_perturb_loss(model, perturbation_info, accelerator, args, mask, logger):
    """
    Compute and back-propagate the perturbation loss.

    For SQuAD: cross-entropy on start/end positions.
    For BoolQ: cross-entropy on the flipped binary label (the perturbed passage
               should push the model toward predicting the *opposite* answer).
    """
    if _is_boolq(args):
        _perturb_loss_boolq(model, perturbation_info, accelerator, args, logger)
    else:
        _perturb_loss_squad(model, perturbation_info, accelerator, args, mask, logger)

def _perturb_loss_boolq(model, perturbation_info, accelerator, args, logger):
    """
    BoolQ perturbation loss.

    Each entry in perturbation_info contains:
        perturbed_batch:  tokenized batch of perturbed passages
        flipped_labels:   1-D LongTensor with the *flipped* gold label
                          (1→0 or 0→1) — the target for the negative example
        mask:             1-D float tensor; 1.0 = use this example, 0.0 = skip

    We apply per-example cross-entropy weighted by the mask.
    """
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for pt_idx in range(args.num_perturbation_examples_per_batch):
        perturbed_batch = perturbation_info[pt_idx]["perturbed_batch"]
        flipped_labels  = perturbation_info[pt_idx]["flipped_labels"]   # LongTensor [B]
        mask            = perturbation_info[pt_idx]["mask"]              # FloatTensor [B]

        outputs = model(**perturbed_batch)
        logits = outputs.logits                          # [B, 2]
        per_example_loss = loss_fct(logits, flipped_labels) * mask

        p_loss = (
            per_example_loss.sum() / mask.sum()
            if mask.sum() > 0
            else 0.0 * per_example_loss.sum()
        )
        p_loss *= args.weight_perturb / args.num_perturbation_examples_per_batch
        accelerator.backward(p_loss)
        logger.info(f"BoolQ perturbed [idx: {pt_idx}] loss: {p_loss.detach().float()}")


def _perturb_loss_squad(model, perturbation_info, accelerator, args, mask, logger):
    for pt_idx in range(args.num_perturbation_examples_per_batch):
        perturbed_batch = perturbation_info[pt_idx]['perturbed_batch']
        p_start_positions = perturbation_info[pt_idx]['p_start_positions']
        p_end_positions = perturbation_info[pt_idx]['p_end_positions']

        if args.no_ans_only:
            no_ans_mask = (p_start_positions == 0) * (p_end_positions == 0)
            mask *= no_ans_mask
        if args.ans_only:
            ans_mask = torch.logical_not((p_start_positions == 0) * (p_end_positions == 0))
            mask *= ans_mask

        mask = perturbation_info[pt_idx]['mask']
        outputs = model(**perturbed_batch)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        ignored_index = start_logits.size(1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index, reduction='none')

        if args.accept_everything_as_negative:
            mask = torch.ones_like(mask)
            p_start_positions = torch.zeros_like(p_start_positions)
            p_end_positions = torch.zeros_like(p_end_positions)

        start_loss = loss_fct(start_logits, p_start_positions) * mask
        end_loss = loss_fct(end_logits, p_end_positions) * mask
        p_loss = (start_loss.sum() + end_loss.sum()) / (2 * mask.sum()) if mask.sum() > 0 else 0.0 * (
                    start_loss.sum() + end_loss.sum())
        p_loss *= args.weight_perturb / args.num_perturbation_examples_per_batch
        accelerator.backward(p_loss)
        logger.info(f"perturbed [idx: {pt_idx}] loss: {p_loss.detach().float()}")


# Permute loss
def calculate_and_backward_permute_loss(model, batch, tokenizer, accelerator, args,
                                        max_seq_length, pad_on_right, logger):
    """
    Compute and back-propagate the permutation loss.

    For SQuAD: shuffle contexts across questions to create no-answer examples.
    For BoolQ: shuffle passages across questions; the permuted pair is always
               a negative (label = 0 = False), since the passage no longer
               supports the original question's answer.
    """
    if _is_boolq(args):
        _permute_loss_boolq(model, batch, tokenizer, accelerator, args, max_seq_length, logger)
    else:
        _permute_loss_squad(model, batch, tokenizer, accelerator, args, max_seq_length, pad_on_right, logger)


def _permute_loss_boolq(model, batch, tokenizer, accelerator, args, max_seq_length, logger):
    """
    BoolQ permutation loss.

    Randomly permute passages across examples in the batch.  Any example
    whose passage is now from a *different* question gets label 0 (False /
    no-longer-answerable).  Examples that happen to receive their own passage
    keep their original label and are excluded from the loss via the mask.
    """
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for pm_idx in range(args.num_permutation_examples_per_batch):
        batch_perm, perm_labels, mask = produce_boolq_perturb_batch(
            batch, tokenizer, args, max_seq_length, logger
        )
        outputs = model(**batch_perm)
        per_example_loss = loss_fct(outputs.logits, perm_labels) * mask
        loss = (
            per_example_loss.sum() / mask.sum()
            if mask.sum() > 0
            else 0.0 * per_example_loss.sum()
        )
        loss *= args.weight_permute / args.num_permutation_examples_per_batch
        accelerator.backward(loss)
        logger.info(f"BoolQ perm [idx: {pm_idx}] loss: {loss.detach().float()}")


def _permute_loss_squad(model, batch, tokenizer, accelerator, args, max_seq_length, pad_on_right, logger):
    for pm_idx in range(args.num_permutation_examples_per_batch):
        batch_perm, mask = produce_no_answer_batch(batch, tokenizer, args, max_seq_length, pad_on_right, logger)
        outputs = model(**batch_perm)

        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        ignored_index = start_logits.size(1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
        p_start_positions = batch_perm['start_positions']
        p_end_positions = batch_perm['end_positions']
        start_loss = loss_fct(start_logits, p_start_positions) * mask
        end_loss = loss_fct(end_logits, p_end_positions) * mask
        loss = (start_loss.sum() + end_loss.sum()) / (2 * mask.sum()) if mask.sum() > 0 else 0.0 * (
                    start_loss.sum() + end_loss.sum())
        loss *= args.weight_permute / args.num_permutation_examples_per_batch
        accelerator.backward(loss)

        logger.info(f"perm [idx: {pm_idx}] loss: {loss.detach().float()}")


# Retrieval loss (SQuAD only)
def calculate_and_backward_retrieval_loss(model, retrieval_dataloader, retrieval_dataloader_iterable, accelerator, args, logger):
    for rt_idx in range(args.num_retrieval):
        try:
            batch_retv = next(retrieval_dataloader_iterable)
        except StopIteration:
            retrieval_dataloader_iterable = iter(retrieval_dataloader)
            batch_retv = next(retrieval_dataloader_iterable)
        outputs = model(**batch_retv)
        rt_loss = outputs.loss
        rt_loss *= args.weight_retrieval / args.num_retrieval
        accelerator.backward(rt_loss)
        logger.info(f"retrieval-based loss: {rt_loss.detach().float()}")
