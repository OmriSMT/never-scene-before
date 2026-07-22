"""
BoolQ mask-strategy additions.

RandomMaskStrategy / POSMaskStrategy / NERMaskStrategy from the original
mask_strategies.py are generic over a list of words and don't touch anything
QA-specific, so they're reused unmodified (imported, not copied).

LossMaskStrategy *is* QA-specific - it needs start/end token positions to
compute a QA loss for each leave-one-out candidate. This file adds a
classification analogue: mask the word whose removal most raises the
sequence classification loss against the BoolQ label.
"""
import torch

from mask_strategies import MaskStrategy, RandomMaskStrategy, POSMaskStrategy, NERMaskStrategy, NewRandomMaskStrategy  # noqa: F401


class ClassificationLossMaskStrategy(MaskStrategy):
    """
    Mask the k question-words whose removal raises the classification loss
    the most (leave-one-out importance), mirroring LossMaskStrategy but using
    a sequence-classification model/loss instead of a QA model/loss.

    Args:
        clf_model: the BoolQ classifier (on correct device, eval mode)
        clf_tokenizer: its tokenizer
        max_seq_length: max tokenizer sequence length
    """

    def __init__(self, clf_model, clf_tokenizer, max_seq_length: int = 384, *args, **kwargs):
        self.model = clf_model
        self.tokenizer = clf_tokenizer
        self.max_seq_length = max_seq_length
        super().__init__(*args, **kwargs)

    def __call__(self, words, context=None, label=None, device=None, **kwargs):
        if context is None:
            raise ValueError("ClassificationLossMaskStrategy requires `context` kwarg")
        if label is None:
            raise ValueError("ClassificationLossMaskStrategy requires `label` kwarg")

        words = [w for w in words if w]
        length = len(words)
        k = int(length * self._batch_prob)
        mask = torch.zeros(1, length, dtype=torch.bool)

        if k <= 0:
            # for efficiency, if k=0 we don't need to compute any losses, just return the empty mask
            return mask

        loss_deltas = []
        label_t = torch.tensor([label], device=device)

        with torch.no_grad():
            for i in range(length):
                candidate = words.copy()
                candidate[i] = "<mask>"
                candidate_q = " ".join(candidate) + "?"

                enc = self.tokenizer(
                    candidate_q,
                    context,
                    truncation="only_second",
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                    padding="max_length",
                ).to(device)

                loss = self.model(**enc, labels=label_t).loss.item()
                loss_deltas.append((i, loss))

        loss_deltas.sort(key=lambda x: x[1], reverse=True)
        top_indices = {idx for idx, _ in loss_deltas[:k]}

        for i in top_indices:
            mask[0, i] = True
        return mask
