from accelerate.logging import get_logger
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import BartForConditionalGeneration, BartTokenizer

logger = get_logger(__name__)


def load_models(args):
    """
    Load the QA model and tokenizer, the BART generator, and the paraphrase classifier.

    Args:
        args: parsed argument namespace

    Returns:
        model: AutoModelForQuestionAnswering
        tokenizer: the QA model tokenizer
        generator_tokenizer: BartForConditionalGeneration used for question perturbation
        tok_gen: BartTokenizer for the generator
        paraphrase_classifier: AutoModelForSequenceClassification trained on QQP
        paraphrase_tokenizer: tokenizer for the paraphrase classifier
    """
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    is_boolq = getattr(args, "dataset_name", "").lower() == "boolq"
    
    print(f"Loading {'BoolQ (SeqCls)' if is_boolq else 'QA'} model {args.model_name_or_path}...")

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if is_boolq:
        # binary classification head
        if args.model_name_or_path:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
        else:
            logger.info("Training new BoolQ model from scratch")
            config.num_labels = 2
            model = AutoModelForSequenceClassification.from_config(config)
        
    else:
        # SQuAD / extractive QA: span-extraction head
        if args.model_name_or_path:
            model = AutoModelForQuestionAnswering.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForQuestionAnswering.from_config(config)

    print("Loading Negative generator...")
    generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    generator = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

    print("Loading Paraphrase classifier...")
    paraphrase_tokenizer = AutoTokenizer.from_pretrained("JeremiahZ/roberta-base-qqp")
    paraphrase_classifier = AutoModelForSequenceClassification.from_pretrained("JeremiahZ/roberta-base-qqp")

    return model, tokenizer, generator, generator_tokenizer, paraphrase_classifier, paraphrase_tokenizer
