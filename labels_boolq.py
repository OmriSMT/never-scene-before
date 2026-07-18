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
