"""Loader for the ACE-whQA evaluation corpus.

ACE-whQA (Sulem, Hay & Roth, "Do We Know What We Don't Know? Studying
Unanswerable Questions beyond SQuAD 2.0", Findings of EMNLP 2021) is an
extractive QA evaluation set distributed in SQuAD 2.0 JSON format. It ships as
three files, each a separate evaluation slice:

    has-answer          -> all questions are answerable
    IDK-competitive     -> unanswerable; passage contains a same-type distractor
    IDK-non-competitive -> unanswerable; passage has no same-type entity

The files live under ``data/ace-whqa/`` and come from the dataset's repository:
https://github.com/CogComp/IDK-beyond-SQuAD2.0/tree/master/DATA/ACE-whQA

Two quirks of the raw files are normalized here so the result matches the
``squad_v2`` schema used elsewhere in this project (e.g. ``eval_checkpoint.py``):
  * ``answer_start`` is stored as a *string* and is cast to ``int``;
  * articles have no ``title`` field, so an empty title is supplied.
"""

import json
import os

import datasets

# Default location the README download snippet writes to.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ace-whqa")

# Friendly split name -> filename on disk.
SPLIT_FILES = {
    "has-answer": "ACE-whQA-has-answer.json",
    "competitive": "ACE-whQA-IDK-competitive.json",
    "non-competitive": "ACE-whQA-IDK-non-competitive.json",
}

# Canonical squad_v2 schema. Declared explicitly so the unanswerable-only files
# (whose answer lists are all empty) get the same int64/string answer types as
# the has-answer file, allowing the splits to be concatenated.
FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "title": datasets.Value("string"),
        "context": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(
            {
                "text": datasets.Value("string"),
                "answer_start": datasets.Value("int32"),
            }
        ),
    }
)


def _read_squad_v2_json(path, id_prefix=""):
    """Flatten a SQuAD 2.0 JSON file into a dict of columns.

    Returns a dict with the canonical columns: ``id``, ``title``, ``context``,
    ``question`` and ``answers`` ({"text": [...], "answer_start": [...]}).
    Unanswerable questions get empty answer lists, matching ``squad_v2``.
    """
    with open(path, "r", encoding="utf-8") as f:
        squad = json.load(f)

    columns = {"id": [], "title": [], "context": [], "question": [], "answers": []}
    for article in squad["data"]:
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                texts = [a["text"] for a in qa["answers"]]
                # answer_start is a string in the raw files; cast to int.
                starts = [int(a["answer_start"]) for a in qa["answers"]]
                columns["id"].append(f"{id_prefix}{qa['id']}")
                columns["title"].append(title)
                columns["context"].append(context)
                columns["question"].append(qa["question"])
                columns["answers"].append({"text": texts, "answer_start": starts})
    return columns


def load_ace_whqa(split, data_dir=DATA_DIR):
    """Load a single ACE-whQA split as a :class:`datasets.Dataset`.

    Args:
        split: one of ``SPLIT_FILES`` ("has-answer", "competitive",
            "non-competitive"), or a path to a SQuAD 2.0 JSON file.
        data_dir: directory containing the ACE-whQA JSON files.

    Returns:
        A ``datasets.Dataset`` with squad_v2-style columns.
    """
    if split in SPLIT_FILES:
        path = os.path.join(data_dir, SPLIT_FILES[split])
    else:
        path = split  # treat as an explicit file path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ACE-whQA file not found: {path}. Download the JSON files from "
            "https://github.com/CogComp/IDK-beyond-SQuAD2.0/tree/master/DATA/ACE-whQA "
            "into data/ace-whqa/."
        )
    return datasets.Dataset.from_dict(_read_squad_v2_json(path), features=FEATURES)


def load_ace_whqa_all(data_dir=DATA_DIR):
    """Load all three splits concatenated into one Dataset.

    Example ids are prefixed with the split name to keep them unique across the
    three files (their local ids otherwise collide, all starting from "0").
    """
    parts = []
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(data_dir, filename)
        columns = _read_squad_v2_json(path, id_prefix=f"{split}-")
        parts.append(datasets.Dataset.from_dict(columns, features=FEATURES))
    return datasets.concatenate_datasets(parts)


if __name__ == "__main__":
    # Smoke test: load each split and print a summary + one sample.
    for name in SPLIT_FILES:
        ds = load_ace_whqa(name)
        n_unans = sum(len(a["text"]) == 0 for a in ds["answers"])
        print(f"{name:16s} {len(ds):4d} examples  ({n_unans} unanswerable)")
    print("\nColumns:", load_ace_whqa("has-answer").column_names)
    print("Sample :", load_ace_whqa("has-answer")[0])