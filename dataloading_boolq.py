from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, default_data_collator

# shahrukhx01/roberta-base-boolq convention: LABEL_0 = "NO" (False), LABEL_1 = "YES" (True)
LABEL2ID = {False: 0, True: 1}


def load_boolq_datasets(args):
    """Load BoolQ train/validation splits from the Hub."""
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    return raw_datasets


def preprocess_boolq(args, raw_datasets, tokenizer, accelerator, max_seq_length):
    """
    Tokenize train/validation splits as (question, passage) -> label, matching
    how shahrukhx01/roberta-base-boolq was originally trained.
    """
    def _prepare(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["passage"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=args.doc_stride,
            padding="max_length" if args.pad_to_max_length else False,
        )
        tokenized["labels"] = [LABEL2ID[bool(a)] for a in examples["answer"]]
        return tokenized

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            _prepare, batched=True, num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Tokenizing BoolQ train split",
        )
        eval_dataset = raw_datasets["validation"].map(
            _prepare, batched=True, num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["validation"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Tokenizing BoolQ validation split",
        )

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    return train_dataset, eval_dataset


def build_dataloaders_boolq(args, train_dataset, eval_dataset, tokenizer, accelerator):
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size, drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size,
    )
    return train_dataloader, eval_dataloader
