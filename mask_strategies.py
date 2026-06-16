import numpy as np
import torch


class RandomMaskStrategy:
    """
    Original behaviour: sample masking probability from Beta(2, 5) once per
    batch, then pick positions uniformly at random.

    The Beta distribution is sampled once when the strategy object is called
    for the first question of a batch and reused for the rest — matching the
    original `mask_questions` behaviour exactly.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 5.0):
        self.alpha = alpha
        self.beta = beta
        self._batch_prob = None   # refreshed at the start of each batch

    def new_batch(self):
        """Call once before processing each question batch."""
        self._batch_prob = np.random.beta(self.alpha, self.beta)

    def __call__(self, words, **kwargs):
        """
        Creating mask.

        Args:
            length: length of array to mask
        """
        # of what proportion the array is masked out
        mask_prob = np.random.beta(2.0, 5.0)
        length = len(words)

        k = int((length - 1) * mask_prob)
        mask = (torch.ones(size=(1, length)) == 0)
        if k > 0:
            rand_mat = torch.rand(1, length - 1)
            k_th_quant = torch.topk(rand_mat, k, largest=False)[0][:, -1:]
            mask[:, :-1] = rand_mat <= k_th_quant
        else:
            mask[:, :-1] = torch.rand(1, length - 1) <= mask_prob
        return mask


class NERMaskStrategy:
    """
    NER-based masking strategy.

    Masks tokens that belong to named entities detected by spaCy.
    The default entity labels focus on factual details that are likely to
    affect answerability in SQuAD-style QA: people, organizations, locations,
    dates, times, and numerical expressions.
    """

    _nlp = None

    def __init__(
        self,
        target_ents=None,
        min_masks=1,
        alpha: float = 2.0,
        beta: float = 5.0,
    ):
        self.target_ents = set(target_ents or [
            "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
            "QUANTITY", "ORDINAL", "CARDINAL"
        ])
        self.min_masks = min_masks
        self.alpha = alpha
        self.beta = beta

        if NERMaskStrategy._nlp is None:
            import spacy
            NERMaskStrategy._nlp = spacy.load(
                "en_core_web_sm",
                disable=["parser"],
            )

    def __call__(self, words, **kwargs):
        device = kwargs.get("device", None)

        length = len(words)
        mask = torch.zeros(size=(1, length), dtype=torch.bool)

        if device is not None:
            mask = mask.to(device)

        if length == 0:
            return mask

        doc = NERMaskStrategy._nlp(" ".join(words))

        candidates = []
        for ent in doc.ents:
            if ent.label_ in self.target_ents:
                for token in ent:
                    if token.i < length:
                        candidates.append(token.i)

        candidates = sorted(set(candidates))

        if len(candidates) == 0:
            return mask

        mask_prob = np.random.beta(self.alpha, self.beta)
        k = int(len(candidates) * mask_prob)

        if self.min_masks is not None:
            k = max(self.min_masks, k)

        k = min(k, len(candidates))

        selected = np.random.choice(candidates, size=k, replace=False)

        for idx in selected:
            mask[0, idx] = True

        return mask
