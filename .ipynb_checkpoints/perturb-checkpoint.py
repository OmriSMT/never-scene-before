import torch 
import numpy as np 
import pandas as pd

from mask_strategies import RandomMaskStrategy


# def mask_questions_and_contexts(questions, contexts):
#     masked_batch = []
#     mask_prob = np.random.choice([0.1, 0.3, 0.5])
#     for q_idx, question in enumerate(questions):
#         question_splits = question.split("?")[0].split(" ")
#         length = len(question_splits)
#         mask = creating_mask(length, mask_prob)
#
#         question_splits_masked = np.array([question_splits], dtype=object)
#         try:
#             question_splits_masked[mask] = "<mask>"
#         except:
#             print(question)
#             print(question_splits)
#             print(mask)
#             print(question_splits_masked)
#
#         question_masked = ' '.join(question_splits_masked[0])
#         question_masked += '?' #question mark
#         question_masked_and_context = question_masked + " " + contexts[q_idx]
#
#         masked_batch.append(question_masked_and_context)
#     return masked_batch


def mask_questions(questions, strategy, contexts=None, start_positions=None, end_positions=None, device=None):
    """
    Mask a batch of questions with a given masking strategy.
    Args:
        questions:       A batch of string questions
        strategy:        a masking strategy instance (RandomMaskStrategy, etc.)
        contexts:        list of context strings, required by LossMaskStrategy
        start_positions: list of int start token positions, required by LossMaskStrategy
        end_positions:   list of int end token positions,   required by LossMaskStrategy
        device:          torch device, passed through to LossMaskStrategy
    
    Output:
        masked_batch: Masked version of input string questions
    """
    masked_batch = []
    for q_idx, question in enumerate(questions):
        words = question.split("?")[0].split(" ")

        extra = {}
        if contexts is not None: extra["context"] = contexts[q_idx]
        if start_positions is not None: extra["start_position"] = start_positions[q_idx]
        if end_positions is not None: extra["end_position"] = end_positions[q_idx]
        if device is not None: extra["device"] = device

        mask = strategy(words, **extra)
        question_splits_masked = np.array([words], dtype=object)
        try:
            question_splits_masked[mask] = "<mask>"
        except:
            print(question)
            print(words)
            print(mask)
            print(question_splits_masked)
                
        question_masked = ' '.join(question_splits_masked[0])
        question_masked += '?' #question mark

        masked_batch.append(question_masked)
    return masked_batch


def mask_passages(passages, strategy, labels=None, device=None):
    """
    Mask a batch of passages (BoolQ) with a given masking strategy.

    Unlike SQuAD, BoolQ has no answer-span boundary — the whole passage is
    the evidence.  We mask at the *word* level (same as mask_questions) so
    that BART can infill plausible substitutions.

    Args:
        passages: list of passage strings
        strategy: masking strategy instance
        labels:   list of int labels (0/1), forwarded to loss-based strategies
        device:   torch device, forwarded to loss-based strategies

    Returns:
        masked_batch: list of masked passage strings
    """
    masked_batch = []
    for p_idx, passage in enumerate(passages):
        words = passage.split(" ")

        extra = {}
        if labels is not None:  extra["label"]  = labels[p_idx]
        if device is not None:  extra["device"] = device

        mask = strategy(words, **extra)
        passage_splits_masked = np.array([words], dtype=object)
        try:
            passage_splits_masked[mask] = "<mask>"
        except Exception:
            print(passage)
            print(words)
            print(mask)
            print(passage_splits_masked)

        masked_passage = " ".join(passage_splits_masked[0])
        masked_batch.append(masked_passage)
    return masked_batch
    

def perturb(batch, tokenizer, generator_tokenizer, generator, tok_para, clf,
        args, max_seq_length, pad_on_right, num_processes = 1, model=None, mask_strategy=None):
    """
    Main SQuAD perturbation functionality. 
    Perturb questions (tokenized by BERT) using BART given contexts
    Args:
        batch: A batch containing input_ids, attention_masks, and ground truth labels
        tokenizer: the roberta/bert tokenizer
        generator_tokenizer: the tokenizer for the genertor
        generator: A generator that takes in masked questions, and fills perturbed questions
        tok_para: the tokenizer for the paraphrase classifier/detector
        clf: A paraphrase detector pretrained on QQP
        args: argparser dictionary
        max_seq_length: maximum sequence length for tokenizers
        pad_on_right: if the padding in tokenizer is to the right
        num_processes: for determining if the training is using multiple GPUs
        mask_strategy:  a masking strategy instance; defaults to RandomMaskStrategy()
    
    Returns:
        perturbed_batch: A batch of perturbed questions
        info: A list of dictionaries containing the original question, masked question, and perturbed question
        success_perturb: A boolean indicating if the perturbation is successful
        mask: A boolean indicating if the perturbation is a paraphrase
    """
    # LossMaskStrategy needs start/end positions; others ignore them.
    start_positions = batch["start_positions"].cpu().tolist()
    end_positions   = batch["end_positions"].cpu().tolist()

    if mask_strategy is None:
        mask_strategy = RandomMaskStrategy()

    device = generator.device 
    original = tokenizer.batch_decode(batch['input_ids'])
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip() for x in original]
    contexts  = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip() for x in original]
    #masked_batch = mask_questions_and_contexts(questions, contexts)
    masked_batch = mask_questions(
        questions,
        strategy=mask_strategy,
        contexts=contexts,
        start_positions=start_positions,
        end_positions=end_positions,
        device=device,
    )    #logger.info(f"masked batch: {masked_batch}")
    input_ids = generator_tokenizer(masked_batch,
                return_tensors="pt", 
                padding=True,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                truncation=True).input_ids

    

    if num_processes > 1:
        generating_func = generator.modules.generate
    else:
        generating_func = generator.generate
    
    
    perturbation = generator_tokenizer.batch_decode(generating_func(
        input_ids.to(device), 
        num_return_sequences=1,
        no_repeat_ngram_size=3, 
        max_length=max_seq_length, 
        do_sample=True, 
        top_p = 0.95, 
        early_stopping=True
    ), skip_special_tokens=True)
    perturbation = [p.split("?")[0].replace('_', '') + '?' for p in perturbation]
    
    info = []
    for q, m, p, c in zip(questions, masked_batch, perturbation, contexts):
        info.append({
            'context': c,
            'question': q,
            'masked_q': m,
            'perturbation': p
        })
        
    success_perturb = True
    
    try:
        tokenized_new_examples = tokenizer(
            perturbation,
            contexts,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length, 
            stride=args.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        ) 

        # Compute mask if the pertubation is a paraphrase 
        # in our setting, 0 : is paraphrase | 1: not paraphrase
        tokenized_pair = tok_para(
            questions,
            perturbation, 
            truncation=True,
            max_length=max_seq_length, 
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length" if args.pad_to_max_length else False,
        )
        
        tokenized_pair_cuda = {}
        for key in tokenized_pair:
            tokenized_pair_cuda[key] = torch.LongTensor(tokenized_pair[key]).to(device)
        clf_output = clf(**tokenized_pair_cuda)
        mask = 1 - torch.argmax(torch.softmax(clf_output.logits, axis=1), axis=1)
    except:
        tokenized_new_examples = batch 
        success_perturb =  False
        mask = torch.zeros(args.per_device_train_batch_size, dtype=torch.long).to(device)

    if success_perturb:
        sep_token_pos_pert = torch.LongTensor([input_ids.index(tokenizer.sep_token_id) for input_ids in tokenized_new_examples['input_ids']]).to(device)
        sep_token_pos_orig = torch.LongTensor([input_ids.cpu().data.numpy().tolist().index(tokenizer.sep_token_id) for input_ids in batch['input_ids']]).to(device)
        perturbed_batch = {}
        perturbed_batch['input_ids'] = torch.LongTensor(tokenized_new_examples['input_ids']).to(device).detach()
        if 'token_type_ids' in tokenized_new_examples:
            perturbed_batch['token_type_ids'] = torch.LongTensor(tokenized_new_examples['token_type_ids']).to(device).detach()
        perturbed_batch['attention_mask'] = torch.LongTensor(tokenized_new_examples['attention_mask']).to(device).detach()

        pos_diff = sep_token_pos_pert - sep_token_pos_orig
        perturbed_batch['start_positions'] = (batch['start_positions'] + pos_diff).to(device)
        perturbed_batch['end_positions'] = (batch['end_positions'] + pos_diff).to(device)

        return perturbed_batch, info, success_perturb, mask
    else:
        info = []
        for q, c in zip(questions, contexts):
            info.append({
                'context': c,
                'question': q,
                'masked_q': q,
                'perturbation': q
            })
        return batch, info, success_perturb, mask


def produce_no_answer_batch(batch, tokenizer, args,
        max_seq_length, pad_on_right, logger, logging=False):
    device = batch['input_ids'].device
    ids = torch.range(0, args.per_device_train_batch_size-1)
    perm_ids = torch.randperm(args.per_device_train_batch_size)
    no_answer_ids = (ids != perm_ids)
    
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    original = tokenizer.batch_decode(batch['input_ids'])
    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip() for x in original]
    contexts  = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip() for x in original]

    perm_contexts = np.array(contexts)[perm_ids].tolist()
    if logging:
        for q, c, gt in zip(questions, perm_contexts, no_answer_ids):
            logger.info("----Permutation Pairs----")
            logger.info(f"Question: {q}")
            logger.info(f"Context:  {c}")
            logger.info(f"NoAnswer: {gt}")

    try:
        tokenized_new_examples = tokenizer(
            questions,
            perm_contexts,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length, 
            stride=args.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        ) 
        batch['start_positions'][no_answer_ids] = 0
        batch['end_positions'][no_answer_ids] = 0
        batch['input_ids'] = torch.LongTensor(tokenized_new_examples['input_ids']).to(device)
        if 'token_type_ids' in tokenized_new_examples:
            batch['token_type_ids'] = torch.LongTensor(tokenized_new_examples['token_type_ids']).to(device)
        batch['attention_mask'] = torch.LongTensor(tokenized_new_examples['attention_mask']).to(device)
        return batch, no_answer_ids.to(device)
    except:
        logger.info('Failed Permutation')
        return batch, torch.zeros_like(no_answer_ids).to(device)


def batch_get_answer_tokens(start_pos, end_pos, input_ids, args):
    batch_answer_tokens = []
    for i in range(args.per_device_train_batch_size):
        if start_pos[i] > end_pos[i]:
            answer_tokens = []
        else:
            answer_tokens = input_ids[i][start_pos[i]:end_pos[i]+1].cpu().data.numpy()
        batch_answer_tokens.append(answer_tokens)
    return batch_answer_tokens


def extract_topk_answer_tokens_from_logits(tokenizer, start_logits, end_logits, input_ids, topk=10, max_answer_length=30):
    """Get topk answer tokens from start_logits and end_logits
    """
    prelim_predictions = []
    null_prediction = {
        "tokens": [tokenizer.cls_token_id],
        "score": (start_logits[0] + end_logits[0]).item(),
        "start_index": 0,
        "end_index": 0
    }
    start_indices = torch.topk(start_logits, k=topk).indices
    end_indices = torch.topk(end_logits, k=topk).indices
    for start_index in start_indices:
        for end_index in end_indices:
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index <= start_index or end_index - start_index + 1 > max_answer_length:
                continue
            tokens = input_ids[start_index:end_index+1].cpu().data.numpy()
            # remove noise from answer extraction
            if tokenizer.sep_token_id in tokens:
                continue
            prelim_predictions.append(
                {
                    "tokens": input_ids[start_index:end_index+1].cpu().data.numpy(),
                    "score": (start_logits[start_index] + end_logits[end_index]).item(),
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )
    prelim_predictions.append(null_prediction)
    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:topk]
    
    return predictions


def batch_get_answer_tokens_topk(tokenizer, start_logits, end_logits, input_ids, args, topk=10, max_answer_length=30):
    """Get topk answer tokens from a batch of start_logits and end_logits
    """
    batch_answer_tokens = []
    for i in range(args.per_device_train_batch_size):
        batch_answer_tokens.append(extract_topk_answer_tokens_from_logits(tokenizer, start_logits[i], end_logits[i], input_ids[i], topk, max_answer_length))
    return batch_answer_tokens


def batch_compute_mIoU(gt_answer_tokens, p_answer_tokens_topk, args, logger, topk=10, method='mean'):
    """Compute mean IoU for a batch of ground truth answer tokens and a batch of predicted answer tokens
    This will be used as a threshold for filtering out bad perturbations
    """
    batch_mIoU = []
    for i in range(args.per_device_train_batch_size):
        answer_tokens = gt_answer_tokens[i]
        list_iou = []
        n_empty = 0
        k = min(len(p_answer_tokens_topk[i]), topk)
        if k < topk:
            logger.info(f"k: {k}")
        for j in range(k):
            p_answer_tokens =  p_answer_tokens_topk[i][j]['tokens']
            if len(p_answer_tokens) == 0: n_empty+=1 
            intersection = np.intersect1d(p_answer_tokens, answer_tokens)
            union = np.union1d(p_answer_tokens, answer_tokens)
            iou = intersection.shape[0] / union.shape[0] if union.shape[0] > 0 else 0.0
            list_iou.append(iou)
        batch_mIoU.append(list_iou)
    return batch_mIoU


def get_topk(retrieved_psgs, index, num_rows, topk = 2):
    retrieved_psgs = np.array(eval(retrieved_psgs))
    retrieved_psgs = retrieved_psgs[retrieved_psgs != index]
    retrieved_psgs = retrieved_psgs[retrieved_psgs < num_rows]
    return retrieved_psgs[:topk]


def flatten_column(df, column_name):
    repeat_lens = [len(item) if item is not np.nan else 1 for item in df[column_name]]
    df_columns = list(df.columns)
    df_columns.remove(column_name)
    expanded_df = pd.DataFrame(np.repeat(df.drop(column_name, axis=1).values, repeat_lens, axis=0), columns=df_columns)
    flat_column_values = np.hstack(df[column_name].values)
    expanded_df[column_name] = flat_column_values
    expanded_df[column_name].replace('nan', np.nan, inplace=True)
    return expanded_df


def evaluate_and_filter_perturbations(
    batch, model, tokenizer, generator_tokenizer, generator,
    paraphrase_tokenizer, paraphrase_classifier, args, max_seq_length, pad_on_right, num_processes, logger
):
    """
    Handles the scouting forward pass, perturbation generation, and
    mIoU / Paraphrase filtering logic. Returns the valid perturbation_info list.
    """
    perturbation_info = []
    with torch.no_grad():
        outputs = model(**batch)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        model_answer_tokens_topk = batch_get_answer_tokens_topk(tokenizer, start_logits, end_logits, batch['input_ids'],
                                                                args)
        model_answer_tokens = [pred[0]['tokens'] for pred in model_answer_tokens_topk]
        m_answers = tokenizer.batch_decode(model_answer_tokens)  # model predictions on original examples

        gt_answer_tokens = batch_get_answer_tokens(batch['start_positions'], batch['end_positions'], batch['input_ids'],
                                                   args)
        gt_answers = tokenizer.batch_decode(gt_answer_tokens)

        for pt_idx in range(args.num_perturbation_examples_per_batch):
            perturbed_batch, info, success_perturb, mask = \
                perturb(batch, tokenizer, generator_tokenizer, generator, paraphrase_tokenizer, paraphrase_classifier, \
                        args, max_seq_length, pad_on_right, num_processes)
            if not args.use_paraphrase_detector:
                mask = torch.ones_like(mask)
            p_outputs = model(**perturbed_batch)  # Model prediction on perturbed examples
            p_start_logits, p_end_logits = p_outputs.start_logits, p_outputs.end_logits
            p_answer_tokens_topk = batch_get_answer_tokens_topk(tokenizer, p_start_logits, p_end_logits,
                                                                perturbed_batch['input_ids'], args)
            batch_mIoU = batch_compute_mIoU(model_answer_tokens, p_answer_tokens_topk, args, logger)
            p_start_positions = torch.zeros(args.per_device_train_batch_size).type(torch.LongTensor).to(model.device)
            p_end_positions = torch.zeros(args.per_device_train_batch_size).type(torch.LongTensor).to(model.device)
            for i in range(args.per_device_train_batch_size):
                example_info = info[i]
                m_answer = m_answers[i]  # model prediction on original example
                g_answer = gt_answers[i]  # groundtruth answer

                pred_topk = p_answer_tokens_topk[i]
                pred_tokens_topk = [pred_topk[j]['tokens'] for j in range(len(pred_topk))]
                p_answers = tokenizer.batch_decode(pred_tokens_topk)  # topk predictions on perturbed example
                IoU_list = batch_mIoU[i]  # IoU between topk predictions and model prediction on original example

                answer_idx = 0
                if mask[i] == 0:  # two sentences are paraphrase
                    logger.info("Perturbation IS a paraphrase")
                    p_answer = p_answers[answer_idx]

                    if (m_answer == g_answer and  # model predicted correctly
                            g_answer == p_answer):  # perturbation didn't change the label
                        logger.info("Robust example")
                        mask[i] = 1  # Robust example, will be kept for training

                else:
                    exists_good_p = False
                    for j in range(len(pred_tokens_topk)):
                        p_answer = p_answers[j]
                        IoU = IoU_list[j]
                        if IoU < args.IoU_threshold:
                            logger.info("Exists perturbation")
                            exists_good_p = True
                            answer_idx = j
                            break
                    p_answer = p_answers[
                        answer_idx]  # best perturbed answer (minimum IoU with model prediction on original example)
                    if not exists_good_p: mask[i] = 0  # Non-paraphrase pertubation didn't change answer

                p_start_positions[i] = pred_topk[answer_idx]['start_index']
                p_end_positions[i] = pred_topk[answer_idx]['end_index']

                success_perturb_i = success_perturb and (example_info['perturbation'] != example_info['question'])
                if (m_answer == g_answer and  # model predicted correctly
                        g_answer == p_answer and  # perturbation didn't change the label
                        success_perturb_i):  # perturbed question is a valid perturbation
                    logger.info("Answer didn't change w.r.t. successful perturbation")
                    mask[i] = 1

                if (tokenizer.cls_token in p_answer and  # perturbed prediction is the same as model prediction
                        tokenizer.cls_token in m_answer and  # both perturbed and orginal predictions are NoAns
                        tokenizer.cls_token not in g_answer):  # groundtruth has answer
                    logger.info("NoAns prediction for both orginal and perturbed. Disregard.")
                    mask[i] = 0

                if not success_perturb_i:
                    logger.info("Unsuccessful perturbation. Disregard.")
                    mask[i] = 0

                do_backprop = mask[
                                  i] > 0.5  # convert mask to boolean. if True, this example will be used for training (via backprop)
                # logger.info(f"context:          {example_info['context']}")
                # logger.info(f"question:         {example_info['question']}")
                # logger.info(f"gt answer:        {g_answer}")
                # logger.info(f"model answer:     {m_answer}")
                # logger.info(f"masked_q:         {example_info['masked_q']}")
                # logger.info(f"perturbation:     {example_info['perturbation']}")
                # logger.info(f"all pert answers: {p_answers}")
                # logger.info(f"topk answer IoU:  {[round(iou, 2) for iou in batch_mIoU[i]]}")
                # logger.info(f"perturbed answer: {p_answer}")
                # logger.info(f"do backprop:      {do_backprop}")
            logger.info(f"mask: {mask}")
            perturbation_info.append({
                'perturbed_batch': perturbed_batch,
                'p_start_positions': p_start_positions,
                'p_end_positions': p_end_positions,
                'mask': mask
            })

    return perturbation_info, mask


# BoolQ perturbation
def perturb_boolq(batch, tokenizer, generator_tokenizer, generator, tok_para, clf,
                  args, max_seq_length, num_processes=1, mask_strategy=None):
    """
    BoolQ perturbation step: mask and infill *passages* (not questions).

    The paraphrase classifier is applied to the (original passage, perturbed passage)
    pair.  A perturbed passage that is NOT a paraphrase is a candidate negative.

    Args:
        batch:               tokenized BoolQ batch (has 'labels' key, no start/end positions)
        tokenizer:           RoBERTa tokenizer
        generator_tokenizer: BART tokenizer
        generator:           BART-large
        tok_para:            paraphrase classifier tokenizer
        clf:                 paraphrase classifier (QQP)
        args:                argument namespace
        max_seq_length:      max token length
        num_processes:       number of GPUs
        mask_strategy:       masking strategy; defaults to RandomMaskStrategy()

    Returns:
        perturbed_batch:  tokenized batch with perturbed passages and *flipped* labels
        info:             list of dicts {question, passage, masked_passage, perturbation}
        success_perturb:  bool
        mask:             LongTensor [B]; 1 = use for training, 0 = skip
    """
    if mask_strategy is None:
        mask_strategy = RandomMaskStrategy()

    device = generator.device
    original = tokenizer.batch_decode(batch["input_ids"])
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    # BoolQ is encoded as [CLS] question [SEP] passage [SEP] (question first, pad_on_right)
    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip()
                 for x in original]
    passages = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip()
                 for x in original]
    labels = batch["labels"].cpu().tolist()

    masked_passages = mask_passages(
        passages,
        strategy=mask_strategy,
        labels=labels,
        device=device,
    )

    input_ids = generator_tokenizer(
        masked_passages,
        return_tensors="pt",
        padding=True,
        max_length=max_seq_length,
        return_overflowing_tokens=False,
        truncation=True,
    ).input_ids

    if num_processes > 1:
        generating_func = generator.module.generate
    else:
        generating_func = generator.generate

    perturbed_passages = generator_tokenizer.batch_decode(
        generating_func(
            input_ids.to(device),
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            max_length=max_seq_length,
            do_sample=True,
            top_p=0.95,
            early_stopping=True,
        ),
        skip_special_tokens=True,
    )

    info = [
        {"question": q, "passage": p, "masked_passage": m, "perturbation": pt}
        for q, p, m, pt in zip(questions, passages, masked_passages, perturbed_passages)
    ]

    success_perturb = True
    try:
        tokenized_new_examples = tokenizer(
            [q.lstrip() for q in questions],
            perturbed_passages,
            truncation=True,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Paraphrase classifier on (original passage, perturbed passage)
        tokenized_pair = tok_para(
            passages,
            perturbed_passages,
            truncation=True,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length" if args.pad_to_max_length else False,
        )
        tokenized_pair_cuda = {k: torch.LongTensor(v).to(device) for k, v in tokenized_pair.items()}
        clf_output = clf(**tokenized_pair_cuda)
        # mask=1 - not paraphrase (good candidate negative)
        # mask=0 - paraphrase (passage barely changed → useless negative)
        mask = 1 - torch.argmax(torch.softmax(clf_output.logits, axis=1), axis=1)

    except Exception:
        tokenized_new_examples = batch
        success_perturb = False
        mask = torch.zeros(args.per_device_train_batch_size, dtype=torch.long).to(device)

    if success_perturb:
        perturbed_batch = {}
        perturbed_batch["input_ids"] = torch.LongTensor(
            tokenized_new_examples["input_ids"]
        ).to(device).detach()
        if "token_type_ids" in tokenized_new_examples:
            perturbed_batch["token_type_ids"] = torch.LongTensor(
                tokenized_new_examples["token_type_ids"]
            ).to(device).detach()
        perturbed_batch["attention_mask"] = torch.LongTensor(
            tokenized_new_examples["attention_mask"]
        ).to(device).detach()
        # target for the perturbed example is the flipped label (True→False or False→True)
        perturbed_batch["labels"] = (1 - batch["labels"]).to(device)

        return perturbed_batch, info, success_perturb, mask
    else:
        perturbed_batch = {k: v.clone() for k, v in batch.items()}
        return perturbed_batch, info, success_perturb, mask


def produce_boolq_perturb_batch(batch, tokenizer, args, max_seq_length, logger, logging=False):
    """
    Create passage-permuted negative examples for BoolQ.

    Randomly shuffles passages across examples in the batch.  Any example
    that received a *different* passage is now a False example (the passage
    no longer supports the original question's true answer), so its target
    label is set to 0.  Examples that happened to receive their own passage
    keep their original label and are excluded from the loss via the mask.

    Args:
        batch:          tokenized BoolQ batch
        tokenizer:      RoBERTa tokenizer
        args:           argument namespace
        max_seq_length: max token length
        logger:         accelerate logger
        logging:        whether to log pairs

    Returns:
        perturbed_batch:  tokenized batch with permuted passages
        perm_labels:      LongTensor [B] target labels (0 for permuted, original for unchanged)
        mask:             FloatTensor [B]; 1.0 = include in loss, 0.0 = exclude
    """
    device = batch["input_ids"].device
    B = args.per_device_train_batch_size

    ids = torch.arange(B)
    perm_ids = torch.randperm(B)
    permuted = (ids != perm_ids)   # True where the passage has been swapped

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    original = tokenizer.batch_decode(batch["input_ids"])
    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip()
                 for x in original]
    passages = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip()
                 for x in original]

    perm_passages = np.array(passages)[perm_ids.numpy()].tolist()
    labels = batch["labels"].cpu().tolist()

    # permuted passages hence label becomes 0 (False); unchanged - keep original label
    perm_labels_list = [0 if permuted[i] else labels[i] for i in range(B)]
    perm_labels = torch.LongTensor(perm_labels_list).to(device)
    mask = permuted.float().to(device)  # only back-prop on actually-permuted examples

    if logging:
        for q, c, gt in zip(questions, perm_passages, permuted):
            logger.info("----BoolQ Permutation Pairs----")
            logger.info(f"Question: {q}")
            logger.info(f"Passage:  {c}")
            logger.info(f"Permuted: {gt}")

    try:
        tokenized_new_examples = tokenizer(
            [q.lstrip() for q in questions],
            perm_passages,
            truncation=True,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length" if args.pad_to_max_length else False,
        )
        perturbed_batch = {}
        perturbed_batch["input_ids"] = torch.LongTensor(
            tokenized_new_examples["input_ids"]
        ).to(device)
        if "token_type_ids" in tokenized_new_examples:
            perturbed_batch["token_type_ids"] = torch.LongTensor(
                tokenized_new_examples["token_type_ids"]
            ).to(device)
        perturbed_batch["attention_mask"] = torch.LongTensor(
            tokenized_new_examples["attention_mask"]
        ).to(device)
        perturbed_batch["labels"] = perm_labels

        return perturbed_batch, perm_labels, mask

    except Exception:
        logger.info("Failed BoolQ Permutation")
        # Return unchanged batch, all labels unchanged, zero mask (no loss contribution)
        return batch, batch["labels"], torch.zeros(B, dtype=torch.float).to(device)

def evaluate_and_filter_perturbations_boolq(
    batch, model, tokenizer, generator_tokenizer, generator,
    paraphrase_tokenizer, paraphrase_classifier, args, max_seq_length,
    num_processes, logger
):
    """
    BoolQ pipeline:
    Scout forward pass → generate passage perturbations → filter by label-flip criterion.

    A perturbed passage is a *useful* negative if:
      1. The perturbation is NOT a paraphrase of the original passage (mask=1 from clf).
      2. The model's prediction on the perturbed passage DIFFERS from its prediction
         on the original passage — i.e., the perturbation actually changed the model output.
      3. The perturbation is different from the original passage (success_perturb=True).

    The target label stored in perturbation_info['flipped_labels'] is always the
    *flipped* gold label (1-original_label), so optimization.py can use it directly
    as the cross-entropy target for training the model to predict the opposite class.

    Returns:
        perturbation_info: list of dicts with keys:
            perturbed_batch, flipped_labels, mask
        mask: final mask from the last perturbation attempt
    """
    perturbation_info = []

    with torch.no_grad():
        outputs = model(**batch)
        original_preds = outputs.logits.argmax(dim=-1) # [B] predicted class on original

        for pt_idx in range(args.num_perturbation_examples_per_batch):
            perturbed_batch, info, success_perturb, mask = perturb_boolq(
                batch, tokenizer, generator_tokenizer, generator,
                paraphrase_tokenizer, paraphrase_classifier,
                args, max_seq_length, num_processes,
            )

            if not args.use_paraphrase_detector:
                mask = torch.ones_like(mask)

            # Model predictions on perturbed passages
            p_outputs  = model(**perturbed_batch)
            perturbed_preds = p_outputs.logits.argmax(dim=-1) # [B]

            gold_labels    = batch["labels"] # [B]  original gold labels
            flipped_labels = (1 - gold_labels).to(model.device) # [B]  flip target

            for i in range(args.per_device_train_batch_size):
                orig_pred = original_preds[i].item()
                pert_pred = perturbed_preds[i].item()
                gold      = gold_labels[i].item()
                same_passage = (info[i]["perturbation"] == info[i]["passage"])

                # filter 1: perturbation failed (BART returned original text)
                if not success_perturb or same_passage:
                    logger.info("BoolQ: unsuccessful perturbation. Disregard.")
                    mask[i] = 0
                    continue

                # filter 2: model was already wrong on original — skip noisy example
                if orig_pred != gold:
                    logger.info("BoolQ: model wrong on original. Disregard.")
                    mask[i] = 0
                    continue

                # filter 3: perturbation didn't change the model prediction — not a useful hard negative
                if pert_pred == orig_pred:
                    logger.info("BoolQ: perturbation did not flip prediction. Disregard.")
                    mask[i] = 0
                    continue

                # survived all filters - makes a valid hard negative
                logger.info(
                    f"BoolQ: valid perturbation "
                    f"(gold={gold}, orig_pred={orig_pred}, pert_pred={pert_pred})"
                )

            logger.info(f"BoolQ mask: {mask}")
            perturbation_info.append({
                "perturbed_batch": perturbed_batch,
                "flipped_labels":  flipped_labels,
                "mask":            mask.float(),
            })

    return perturbation_info, mask