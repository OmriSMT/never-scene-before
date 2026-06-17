from abc import ABC, abstractmethod
import numpy as np
import torch


class MaskStrategy(ABC):
    """
    Base class for masking strategies.  Subclasses should implement the __call__ method.
    """
    def __init__(self, alpha: float=2.0, beta: float=5.0):
        self.alpha = alpha
        self.beta = beta
        self._batch_prob: float | None = None  # refreshed at the start of each batch

    def sample_mask_proportion(self):
        """Call once before processing each question batch."""
        self._batch_prob = np.random.beta(self.alpha, self.beta)

    @abstractmethod
    def __call__(self, words, **kwargs):
        """
        Create a boolean mask for the given list of words.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class RandomMaskStrategy(MaskStrategy):
    """
    Original behaviour: sample masking probability from Beta(2, 5) once per
    batch, then pick positions uniformly at random.

    The Beta distribution is sampled once when the strategy object is called
    for the first question of a batch and reused for the rest — matching the
    original `mask_questions` behaviour exactly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, words, **kwargs):
        """
        Creating mask.

        Args:
            length: length of array to mask
        """
        length = len(words)

        # of what proportion the array is masked out
        k = int(length * self._batch_prob)

        mask = (torch.ones(size=(1, length)) == 0)
        if k > 0:
            rand_mat = torch.rand(1, length - 1)
            k_th_quant = torch.topk(rand_mat, k, largest=False)[0][:, -1:]
            mask[:, :-1] = rand_mat <= k_th_quant
        else:
            mask[:, :-1] = torch.rand(1, length - 1) <= self._batch_prob
        return mask


class NERMaskStrategy(MaskStrategy):
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
        *args,
        **kwargs,
    ):
        self.target_ents = set(target_ents or [
            "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
            "QUANTITY", "ORDINAL", "CARDINAL"
        ])
        if NERMaskStrategy._nlp is None:
            import spacy
            NERMaskStrategy._nlp = spacy.load(
                "en_core_web_sm",
                disable=["parser"],
            )
        super().__init__(*args, **kwargs)

    def __call__(self, words, **kwargs):
        length = len(words)
        mask = torch.zeros(size=(1, length), dtype=torch.bool)

        if length == 0:
            return mask

        doc = NERMaskStrategy._nlp(" ".join(words))

        # build char_index → word_index map
        # This avoids bugs when spaCy tokenization differs from the words list,
        # for example: "didn't" -> "did" + "n't".
        char_to_word = {}
        char = 0
        for word_idx, word in enumerate(words):
            for c in range(char, char + len(word)):
                char_to_word[c] = word_idx
            char += len(word) + 1  # +1 for the space between words

        candidates = []
        for ent in doc.ents:
            if ent.label_ in self.target_ents:
                for token in ent:
                    word_idx = char_to_word.get(token.idx)
                    if word_idx is not None and word_idx < length:
                        candidates.append(word_idx)

        candidates = sorted(set(candidates))

        if len(candidates) == 0:
            return mask

        k = int(length * self._batch_prob)
        k = min(k, len(candidates)) # ensure k does not exceed number of candidates

        selected = np.random.choice(candidates, size=k, replace=False)

        for idx in selected:
            mask[0, idx] = True

        return mask
