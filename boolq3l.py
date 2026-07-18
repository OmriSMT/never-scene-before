"""Loader for the BoolQ3L (BoolQ with three labels) evaluation corpus.

BoolQ3L (Sulem, Hay & Roth, "Yes, No or IDK: The Challenge of Unanswerable
Yes/No Questions", NAACL 2022) extends BoolQ with unanswerable ("IDK")
questions, giving a three-way label space instead of BoolQ's binary yes/no.
This makes it the natural evaluation set for the SCENE BoolQ classifier, whose
head predicts NO / YES / NO ANSWER.

The files come from the dataset's repository:
https://github.com/CogComp/Yes-No-or-IDK/tree/main/DATA/BoolQ_3L
and live under ``data/boolq-3l/`` (git-ignored). Each line is a JSON object with
fields ``question``, ``title``, ``passage`` and ``answer``, where ``answer`` is
``true`` (YES), ``false`` (NO) or the string ``"no-answer"`` (IDK). The dev
split has 4,906 examples, ~33% of them IDK.

The raw ``answer`` field is normalized here to the integer label ids used
throughout this project (see ``labels_boolq.LABEL2ID``), so the result feeds
straight into the BoolQ classifier without any binary-only assumptions.
"""

import json
import os

import datasets

from labels_boolq import ID2LABEL, LABEL2ID

# Default location the data was downloaded to.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "boolq-3l")

# Friendly split name -> filename on disk.
SPLIT_FILES = {
    "dev": "dev_full.jsonl",
    "train": "train_full.jsonl",
}

# Canonical schema: (question, passage) -> 3-way label, matching how the BoolQ
# classifier is tokenized/trained (see dataloading_boolq.preprocess_boolq).
FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "title": datasets.Value("string"),
        "question": datasets.Value("string"),
        "passage": datasets.Value("string"),
        "label": datasets.ClassLabel(names=[ID2LABEL[i] for i in sorted(ID2LABEL)]),
    }
)

# Strings the raw ``answer`` field uses for the unanswerable class.
_IDK_STRINGS = {"no-answer", "no answer", "idk"}


def _answer_to_label(answer):
    """Map a raw BoolQ3L ``answer`` value to a label id (0=NO, 1=YES, 2=NO ANSWER)."""
    if isinstance(answer, str) and answer.strip().lower() in _IDK_STRINGS:
        return LABEL2ID["NO ANSWER"]
    return LABEL2ID["YES"] if bool(answer) else LABEL2ID["NO"]


def _read_jsonl(path, id_prefix=""):
    """Flatten a BoolQ3L JSONL file into a dict of columns."""
    columns = {"id": [], "title": [], "question": [], "passage": [], "label": []}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            columns["id"].append(f"{id_prefix}{i}")
            columns["title"].append(record.get("title", "") or "")
            columns["question"].append(record["question"])
            columns["passage"].append(record["passage"])
            columns["label"].append(_answer_to_label(record["answer"]))
    return columns


def load_boolq3l(split="dev", data_dir=DATA_DIR):
    """Load a BoolQ3L split as a :class:`datasets.Dataset`.

    Args:
        split: one of ``SPLIT_FILES`` ("dev", "train"), or a path to a
            BoolQ3L JSONL file.
        data_dir: directory containing the BoolQ3L files.

    Returns:
        A ``datasets.Dataset`` with columns ``id``, ``title``, ``question``,
        ``passage`` and ``label`` (0=NO, 1=YES, 2=NO ANSWER).
    """
    if split in SPLIT_FILES:
        path = os.path.join(data_dir, SPLIT_FILES[split])
    else:
        path = split  # treat as an explicit file path
    _require(path)
    return datasets.Dataset.from_dict(_read_jsonl(path), features=FEATURES)


def load_boolq3l_all(data_dir=DATA_DIR):
    """Load all BoolQ3L splits concatenated into one Dataset.

    Example ids are prefixed with the split name to keep them unique across the
    files (their local ids otherwise collide, both starting from "0").
    """
    parts = []
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(data_dir, filename)
        _require(path)
        columns = _read_jsonl(path, id_prefix=f"{split}-")
        parts.append(datasets.Dataset.from_dict(columns, features=FEATURES))
    return datasets.concatenate_datasets(parts)


def _require(path):
    """Raise a helpful error if a BoolQ3L file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BoolQ3L file not found: {path}. Download the JSON files from "
            "https://github.com/CogComp/Yes-No-or-IDK/tree/main/DATA/BoolQ_3L "
            "into data/boolq-3l/ (dev_full.jsonl / train_full.jsonl)."
        )


if __name__ == "__main__":
    # Smoke test: load each available split and print a summary + one sample.
    from collections import Counter

    for name in SPLIT_FILES:
        try:
            ds = load_boolq3l(name)
        except FileNotFoundError as e:
            print(f"{name:5s} (skipped: {e})")
            continue
        counts = Counter(ID2LABEL[l] for l in ds["label"])
        print(f"{name:5s} {len(ds)} examples  label counts: {dict(counts)}")

    ds = load_boolq3l("dev")
    print("Columns:", ds.column_names)
    print("Sample :", {k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                        for k, v in ds[0].items()})