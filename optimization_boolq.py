import math

import torch
from transformers import get_scheduler


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
