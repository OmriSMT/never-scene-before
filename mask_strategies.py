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
