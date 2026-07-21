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


class NewRandomMaskStrategy(MaskStrategy):
    """
    This fixed random does not mask tokens even if k=0, like in the original.
    This is to be more consistent with the other strategies.
    """
    def __call__(self, words, **kwargs):
        length = len(words)
        k = int(length * self._batch_prob)

        mask = torch.zeros(1, length, dtype=torch.bool)
        if k > 0:
            selected = np.random.choice(length, size=k, replace=False)
            for idx in selected:
                mask[0, idx] = True
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
        if start_position is None or end_position is None:
            raise ValueError("LossMaskStrategy requires `start_position` and `end_position` kwargs")

        # strip empty strings from multi-space splits to avoid malformed candidates
        words = [w for w in words if w]
        length = len(words)
        # of what proportion the array is masked out
        k = int(length * self._batch_prob)
        mask = torch.zeros(1, length, dtype=torch.bool)

        if k <= 0 or length == 0:
            # for efficiency, if k=0 we don't need to compute any losses, just return the empty mask
            return mask

        loss_deltas = []

        # Encode the original question once to derive character-level answer
        # span — this is the same for every leave-one-out variant.
        original_q = " ".join(words) + "?"
        original_enc = self.tokenizer(
            original_q,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = original_enc["offset_mapping"][0]
        start_char = offsets[start_position][0].item() if start_position < len(offsets) else 0
        end_char = offsets[end_position][1].item() if end_position < len(offsets) else 0

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

                offsets_new = enc["offset_mapping"][0]
                seq_ids_new = enc.sequence_ids(0)

                # re-derive token start/end from character offsets in the new encoding
                new_start, new_end = 0, 0  # default to CLS (no-answer) if not found
                for j, (off, sid) in enumerate(zip(offsets_new, seq_ids_new)):
                    if sid != 1:
                        continue
                    if off[0].item() <= start_char < off[1].item():
                        new_start = j
                    if off[0].item() < end_char <= off[1].item():
                        new_end = j

                enc.pop("offset_mapping")  # model doesn't accept this field

                loss = self.model(
                    **enc,
                    start_positions=torch.tensor([new_start], device=device),
                    end_positions=torch.tensor([new_end], device=device),
                ).loss.item()
                loss_deltas.append((i, loss))

        # pick the k positions with the highest loss
        loss_deltas.sort(key=lambda x: x[1], reverse=True)
        top_indices = {idx for idx, _ in loss_deltas[:k]}

        for i in top_indices:
            mask[0, i] = True
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
        k = int(length * self._batch_prob)

        if k == 0 or length == 0:
            # for efficiency, if k=0 we don't need to compute anything, just return the empty mask
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

        k = min(k, len(candidates)) # ensure k does not exceed number of candidates
        if k == 0:
            return mask

        selected = np.random.choice(candidates, size=k, replace=False)

        for idx in selected:
            mask[0, idx] = True

        return mask


class POSMaskStrategy(MaskStrategy):
    """
    POS-based masking strategy.

    Masks only tokens whose POS tag is in target_pos.
    For example: NOUN, PROPN, VERB, ADJ, NUM.
    """

    _nlp = None

    def __init__(
            self,
            target_pos=("NOUN", "PROPN", "VERB", "ADJ", "NUM"),
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.target_pos = set(target_pos)

        if POSMaskStrategy._nlp is None:
            import spacy
            POSMaskStrategy._nlp = spacy.load(
                "en_core_web_sm",
                disable=["ner", "parser"],
            )

    def __call__(self, words, **kwargs):
        device = kwargs.get("device", None)

        length = len(words)
        mask = torch.zeros(size=(1, length), dtype=torch.bool)
        k = int(length * self._batch_prob)

        if k == 0 or length == 0:
            # for efficiency, if k=0 we don't need to compute anything, just return the empty mask
            return mask

        if device is not None:
            mask = mask.to(device)

        text = " ".join(words)
        doc = POSMaskStrategy._nlp(text)

        # Build char offset -> original word index map.
        # This avoids bugs when spaCy tokenization differs from the words list,
        # for example: "didn't" -> "did" + "n't".
        char_to_word = {}
        char = 0
        for word_idx, word in enumerate(words):
            for c in range(char, char + len(word)):
                char_to_word[c] = word_idx
            char += len(word) + 1  # +1 for the space

        candidates = []
        for token in doc:
            if token.pos_ in self.target_pos:
                word_idx = char_to_word.get(token.idx)
                if word_idx is not None and word_idx < length:
                    candidates.append(word_idx)

        candidates = sorted(set(candidates))

        if self._batch_prob is None:
            self.sample_mask_proportion()

        k = min(k, len(candidates))

        if k == 0:
            return mask

        selected = np.random.choice(candidates, size=k, replace=False)

        for idx in selected:
            mask[0, idx] = True

        return mask
