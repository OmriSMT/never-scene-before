import math


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

    return optimizer, optim_gen, lr_scheduler, max_train_steps


def calculate_and_backward_perturb_loss(model, perturbation_info, accelerator, args, mask):
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


def calculate_and_backward_permute_loss(model, batch, tokenizer, accelerator, args, max_seq_length, pad_on_right, logger):
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


def calculate_and_backward_retrieval_loss(model, retrieval_dataloader_iterable, accelerator, args):
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
