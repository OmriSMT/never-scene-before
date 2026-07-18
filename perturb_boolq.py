"""
BoolQ adaptation of perturb.py's SCENE machinery.

keeped:
  1. mask some question words (RandomMaskStrategy / POSMaskStrategy /
     NERMaskStrategy / ClassificationLossMaskStrategy)
  2. have BART regenerate the masked question
  3. use a QQP paraphrase classifier to tell whether the regenerated
     question still means the same thing

Removed from og file (no BoolQ equivalent):
  - offset_mapping / doc_stride bookkeeping (no answer spans to track)
  - IoU-based span-overlap filtering -> replaced with simple label-equality
    checks, since BoolQ labels are a single True/False, not a span
  - self-labeling target: SCENE self-labels perturbed examples with the
    model's own *predicted answer span*; here we self-label with the
    model's own *predicted class* on the perturbed (question', passage) pair
"""
import numpy as np
import torch
import torch.nn.functional as F

from perturb import mask_questions
from mask_strategies_boolq import ClassificationLossMaskStrategy


def perturb_boolq(batch, tokenizer, generator_tokenizer, generator,
                   args, max_seq_length, mask_strategy, num_processes=1):
    """
    Mask + regenerate the question half of a tokenized BoolQ batch.

    Args:
        batch: dict with input_ids/attention_mask/labels, tokenized as
               tokenizer(question, passage, ...) -- question first, matching
               shahrukhx01/roberta-base-boolq's training format.
        tokenizer: the BoolQ classifier's tokenizer
        generator_tokenizer / generator: BART used to fill masked questions
        args: parsed argument namespace
        max_seq_length: effective max sequence length
        mask_strategy: a MaskStrategy instance
        num_processes: >1 under multi-GPU DDP (affects generator.generate access)

    Returns:
        perturbed_batch: dict of input_ids/attention_mask/labels (labels
                          copied unchanged from `batch` -- caller overwrites
                          them with the self-labeling decision)
        info: list of dicts with the original/masked/perturbed question text
        success_perturb: bool, whether generation produced *some* output
    """
    device = generator.device
    labels = batch["labels"].cpu().tolist()
    original = tokenizer.batch_decode(batch["input_ids"])
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip() for x in original]
    passages = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip() for x in original]

    masked_batch = mask_questions(
        questions,
        strategy=mask_strategy,
        contexts=passages,        # harmless extra kwarg for strategies that don't need it
        start_positions=None,
        end_positions=None,
        device=device,
    )
    # ClassificationLossMaskStrategy needs `passage`/`label` per example, which
    # mask_questions() doesn't know how to pass through -- handle it directly.
    if isinstance(mask_strategy, ClassificationLossMaskStrategy):
        mask_strategy.sample_mask_proportion()
        masked_batch = []
        for q, p, y in zip(questions, passages, labels):
            words = [w for w in q.split("?")[0].split(" ") if w]
            mask = mask_strategy(words, passage=p, label=y, device=device)
            words_arr = np.array([words], dtype=object)
            words_arr[mask] = "<mask>"
            masked_batch.append(" ".join(words_arr[0]) + "?")

    gen_input_ids = generator_tokenizer(
        masked_batch,
        return_tensors="pt",
        padding=True,
        max_length=max_seq_length,
        truncation=True,
    ).input_ids

    generating_func = generator.module.generate if num_processes > 1 else generator.generate
    perturbation = generator_tokenizer.batch_decode(
        generating_func(
            gen_input_ids.to(device),
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            max_length=max_seq_length,
            do_sample=True,
            top_p=0.95,
            early_stopping=True,
        ),
        skip_special_tokens=True,
    )
    perturbation = [p.split("?")[0].replace("_", "") + "?" for p in perturbation]

    info = [
        {"passage": c, "question": q, "masked_q": m, "perturbation": p}
        for q, m, p, c in zip(questions, masked_batch, perturbation, passages)
    ]

    try:
        tokenized_new = tokenizer(
            perturbation,
            passages,
            truncation="only_second",
            max_length=max_seq_length,
            padding="max_length" if args.pad_to_max_length else True,
            return_tensors="pt",
        )
        success_perturb = True
    except Exception:
        tokenized_new = tokenizer(
            questions,
            passages,
            truncation="only_second",
            max_length=max_seq_length,
            padding="max_length" if args.pad_to_max_length else True,
            return_tensors="pt",
        )
        success_perturb = False

    perturbed_batch = {
        "input_ids": tokenized_new["input_ids"].to(device),
        "attention_mask": tokenized_new["attention_mask"].to(device),
        "labels": batch["labels"].clone().to(device),  # placeholder, caller overwrites
    }
    return perturbed_batch, info, success_perturb


def evaluate_and_filter_perturbations_boolq(
    batch, model, tokenizer, generator_tokenizer, generator,
    paraphrase_tokenizer, paraphrase_classifier,
    args, max_seq_length, mask_strategy, logger, num_processes=1,
):
    """
    Scouting forward pass + perturbation generation + self-labeling filter.

    Self-labeling rule (see module docstring for the SCENE -> BoolQ mapping):
      - paraphrase (meaning preserved): keep iff model already gets both the
        original and perturbed question right; pseudo-label = ground truth
      - not a paraphrase (meaning changed): keep iff model got the original
        right; pseudo-label = model's own prediction on the perturbed input
        (self-labeled counterfactual)

    Returns:
        perturbed_batch: dict of input_ids/attention_mask ready for model(**batch)
        pseudo_labels: LongTensor[batch] the self-labeled targets
        keep_mask: FloatTensor[batch], 1.0 for examples to backprop on, else 0.0
        info: list of per-example dicts (for logging/inspection)
    """
    with torch.no_grad():
        orig_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        pred_orig = orig_logits.argmax(dim=-1)
        gt = batch["labels"]

        perturbed_batch, info, success_perturb = perturb_boolq(
            batch, tokenizer, generator_tokenizer, generator,
            args, max_seq_length, mask_strategy, num_processes=num_processes,
        )

        # Paraphrase check via QQP classifier: label 1 = paraphrase (same
        # meaning), matching the convention used in the original perturb.py.
        questions = [d["question"] for d in info]
        perturbations = [d["perturbation"] for d in info]
        tokenized_pair = paraphrase_tokenizer(
            questions, perturbations, truncation=True, max_length=128,
            padding=True, return_tensors="pt",
        ).to(model.device)
        qqp_logits = paraphrase_classifier(**tokenized_pair).logits
        is_paraphrase = torch.argmax(F.softmax(qqp_logits, dim=-1), dim=-1)  # 1 = paraphrase
        not_paraphrase = (1 - is_paraphrase).bool()

        pert_logits = model(input_ids=perturbed_batch["input_ids"],
                             attention_mask=perturbed_batch["attention_mask"]).logits
        pred_pert = pert_logits.argmax(dim=-1)

        orig_correct = (pred_orig == gt)
        fill_value = True if success_perturb else False
        success_perturb_t = torch.full_like(gt, fill_value=fill_value, dtype=torch.bool)

        pseudo_labels = gt.clone()
        keep_mask = torch.zeros_like(gt, dtype=torch.float)

        is_robust_case = (~not_paraphrase) & success_perturb_t
        keep_robust = is_robust_case & orig_correct & (pred_pert == gt)
        pseudo_labels = torch.where(keep_robust, gt, pseudo_labels)
        keep_mask = torch.where(keep_robust, torch.ones_like(keep_mask), keep_mask)

        is_counterfactual_case = not_paraphrase & success_perturb_t
        keep_counterfactual = is_counterfactual_case & orig_correct
        pseudo_labels = torch.where(keep_counterfactual, pred_pert, pseudo_labels)
        keep_mask = torch.where(keep_counterfactual, torch.ones_like(keep_mask), keep_mask)

        n_robust = int(keep_robust.sum().item())
        n_counterfactual = int(keep_counterfactual.sum().item())
        logger.info(f"perturb_boolq: kept {n_robust} robust-paraphrase + {n_counterfactual} counterfactual "
                    f"of {gt.numel()} candidates")

    return perturbed_batch, pseudo_labels, keep_mask, info
