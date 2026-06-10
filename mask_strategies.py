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
        k = int(length * self._batch_prob)  # last word is "?"

        mask = (torch.ones(size=(1, length)) == 0)
        if k > 0:
            rand_mat = torch.rand(1, length - 1)
            k_th_quant = torch.topk(rand_mat, k, largest=False)[0][:, -1:]
            mask[:, :-1] = rand_mat <= k_th_quant
        else:
            mask[:, :-1] = torch.rand(1, length - 1) <= self._batch_prob
        return mask


class LossMaskStrategy(MaskStrategy):
    """
    Mask the k words whose removal raises the QA loss the most
    (leave-one-out importance).

    For each word position i (excluding "?"), builds a version of the question
    with word i replaced by <mask>, feeds it through the QA model, and records
    the loss.  The k positions with the highest loss are selected for masking.

    This is the straightforward serial implementation: N forward passes for a
    question with N words.

    Args:
        qa_model:       QA model (on correct device, in eval mode)
        qa_tokenizer:   RoBERTa/BERT tokenizer (not BART)
        k:              number of words to mask
        max_seq_length: max tokenizer sequence length
    """

    def __init__(
            self,
            qa_model,
            qa_tokenizer,
            max_seq_length: int = 384,
            *args,
            **kwargs,
    ):
        self.model = qa_model
        self.tokenizer = qa_tokenizer
        self.max_seq_length = max_seq_length
        super().__init__(*args, **kwargs)


    def __call__(self, words, context=None, start_position=None,
                 end_position=None, device=None, **kwargs):
        if context is None:
            raise ValueError("LossMaskStrategy requires `context` kwarg")

        length = len(words)

        # of what proportion the array is masked out
        k = int(length  * self._batch_prob)


        device = device or next(self.model.parameters()).device
        loss_deltas = []

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
                    return_offsets_mapping=True,
                ).to(device)

                enc["start_positions"] = torch.tensor([start_position], device=device)
                enc["end_positions"] = torch.tensor([end_position], device=device)

                loss = self.model(**enc).loss.item()
                loss_deltas.append((i, loss))

        # pick the k positions with the highest loss
        loss_deltas.sort(key=lambda x: x[1], reverse=True)
        top_indices = {idx for idx, _ in loss_deltas[:k]}

        mask = torch.zeros(1, length, dtype=torch.bool)
        for i in top_indices:
            mask[0, i] = True
        return mask
