"""
Fine-tune roberta-base on google/boolq using a manual `accelerate` training
loop, built on top of the functions already in dataloading_boolq.py
(load_boolq_datasets / preprocess_boolq / build_dataloaders_boolq).

Usage:
    accelerate launch train_boolq_base.py \
        --dataset_name google/boolq \
        --model_name_or_path roberta-base \
        --max_seq_length 384 \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 16 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --output_dir data/roberta-base-boolq-accelerate
"""

import argparse
import math

import evaluate
import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)


from dataloading_boolq import (
    load_boolq_datasets,
    preprocess_boolq,
    build_dataloaders_boolq,
)


"""
Boolq dataset labels for training
"""
LABEL2ID = {
    "NO": 0,
    "YES": 1,
    "NO ANSWER": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NUM_LABELS = len(LABEL2ID)


def parse_args():
    parser = argparse.ArgumentParser(description="Accelerate training loop for RoBERTa on BoolQ")

    # data
    parser.add_argument("--dataset_name", type=str, default="google/boolq")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128,
                        help="Unused for classification but kept since dataloading_boolq.py expects it.")
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    # model
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")

    # optimization
    parser.add_argument("--per_device_train_batch_size", type=int, default=12)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="If set, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    # misc
    parser.add_argument("--output_dir", type=str, default="data/roberta-base-boolq-accelerate")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator()
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # --- reuse dataloading_boolq.py exactly as-is ---
    raw_datasets = load_boolq_datasets(args)
    train_dataset, eval_dataset = preprocess_boolq(
        args, raw_datasets, tokenizer, accelerator, args.max_seq_length
    )
    train_dataloader, eval_dataloader = build_dataloaders_boolq(
        args, train_dataset, eval_dataset, tokenizer, accelerator
    )

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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(math.ceil(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # --- eval after each epoch ---
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
        acc = accuracy_metric.compute(predictions=all_preds, references=all_refs)
        f1 = f1_metric.compute(predictions=all_preds, references=all_refs, average="macro")

        if accelerator.is_local_main_process:
            print(f"epoch {epoch}: accuracy={acc['accuracy']:.4f} f1_macro={f1['f1']:.4f}")

    # --- save ---
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model + tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
