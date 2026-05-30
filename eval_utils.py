

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def post_processing_function(examples, features, predictions, args, stage="eval"):
    """
    Post-processing: we match the start logits and end logits to answers in the original context.
    """
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=args.version_2_with_negative,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        #null_score_diff_threshold=args.null_score_diff_threshold,
        no_answer_probability_threshold=args.no_answer_probability_threshold,
        without_threshold=(not args.use_threshold),
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": prob} for k, (v, prob) in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, (v, _) in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def run_evaluation(model, dataloader, dataset, examples, accelerator, metric, args, is_test=False):
    all_start_logits = []
    all_end_logits = []

    model.eval()

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(examples, dataset, outputs_numpy, args)
    metrics = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    if is_test:
        logger.info(f"Test metrics: {metrics}")
    else:
        logger.info(f"Validation metrics: {metrics}")

    return  metrics
