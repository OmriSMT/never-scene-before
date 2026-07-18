from accelerate.logging import get_logger
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

logger = get_logger(__name__)


def load_models_boolq(args):
    """
    Load the BoolQ classifier + tokenizer, the BART generator used to
    regenerate masked questions, and the QQP paraphrase classifier used to
    judge whether a regenerated question changed meaning.

    Mirrors models.load_models(), with AutoModelForQuestionAnswering swapped
    for AutoModelForSequenceClassification.
    """
    print(f"Loading BoolQ classifier {args.model_name_or_path}...")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    print("Loading negative-question generator (BART)...")
    generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    generator = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

    print("Loading paraphrase classifier (QQP)...")
    paraphrase_tokenizer = AutoTokenizer.from_pretrained("JeremiahZ/roberta-base-qqp")
    paraphrase_classifier = AutoModelForSequenceClassification.from_pretrained("JeremiahZ/roberta-base-qqp")

    return model, tokenizer, generator, generator_tokenizer, paraphrase_classifier, paraphrase_tokenizer
