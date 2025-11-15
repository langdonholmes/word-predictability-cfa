import torch
import numpy as np
from itertools import tee

from dataclasses import dataclass, field
from typing import Optional
from spacy.tokens import Doc
import numpy.typing as npt

crossloss = torch.nn.CrossEntropyLoss(reduction="none")


def pairwise(arr, add_extra=True) -> zip:
    """
    [0, 1, 3, 4] --> [(0,1), (1,3), (3,4)]
    Add an extra value to end of iterable, so
    [0, 1, 3, 4] --> [0, 1, 3, 4, 5] --> [(0,1), (1,3), (3,4), (4,5)]
    """
    if add_extra:
        arr = np.append(arr, arr[-1] + 1)
    a, b = tee(arr)
    next(b, None)
    return zip(a, b)


@dataclass
class TokenPredictability:
    """
    Predictability metrics for a single spaCy token.
    """

    spacy_idx: int
    text: str

    # Per-subword metrics
    subword_losses: list[float]  # AKA surprisals
    subword_entropy: list[float]  # Entropy of predicted distribution
    hidden_state_pred: list[npt.NDArray] = field(default_factory=list)
    hidden_state_obs: list[npt.NDArray] = field(default_factory=list)

    # Computed fields (derived in __post_init__)
    subword_probs: list[float] = field(init=False)
    mean_prob: float = field(init=False)
    mean_loss: float = field(init=False)
    mean_entropy: float = field(init=False)

    def __post_init__(self):
        """Calculate aggregated metrics."""
        if self.subword_losses:
            # Derive probabilities from losses: p = exp(-loss)
            self.subword_probs = [np.exp(-loss) for loss in self.subword_losses]
            self.mean_prob = np.mean(self.subword_probs)
            self.mean_loss = np.mean(self.subword_losses)  # Average surprisal
            self.mean_entropy = (
                np.mean(self.subword_entropy) if self.subword_entropy else 0.0
            )
        else:
            self.mean_prob = 0.0
            self.mean_loss = float("inf")
            self.mean_entropy = 0.0

    def __len__(self) -> int:
        """Number of subwords for this token."""
        return len(self.subword_probs)

    def to_dict(self) -> dict:
        """Convert to dictionary for easy serialization."""
        return {
            "spacy_idx": self.spacy_idx,
            "text": self.text,
            "subword_probs": self.subword_probs,
            "subword_losses": self.subword_losses,
            "subword_entropy": self.subword_entropy,
            "mean_prob": self.mean_prob,
            "mean_loss": self.mean_loss,
            "mean_entropy": self.mean_entropy,
            "num_subwords": len(self),
        }


def get_centered_window(
    seq_len: int, center_idx: int, window_size: int
) -> tuple[int, int]:
    """
    Get a window centered on center_idx.
    If center_idx is near start/end, window extends further in the other direction.

    Returns (start, end) indices for the window.
    """
    half_window = window_size // 2

    # Ideal centered window
    start = center_idx - half_window
    end = center_idx + half_window

    # Adjust if we're near the boundaries
    if start < 0:
        # Near the start - extend to the right
        end = min(window_size, seq_len)
        start = 0
    elif end > seq_len:
        # Near the end - extend to the left
        start = max(0, seq_len - window_size)
        end = seq_len

    return start, end


@dataclass
class DocPredictability:
    """
    Collection of predictability metrics for all tokens in a document.
    """

    tokens: list[TokenPredictability] = field(default_factory=list)
    model_type: str = ""  # 'masked' or 'causal'
    window_size: Optional[int] = None

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> TokenPredictability:
        """Access token by index."""
        return self.tokens[idx]

    def __iter__(self):
        """Iterate over tokens."""
        return iter(self.tokens)

    def add_token(self, token_pred: TokenPredictability):
        """Add a token's predictability results."""
        self.tokens.append(token_pred)

    def get_by_spacy_idx(self, spacy_idx: int) -> Optional[TokenPredictability]:
        """Get token predictability by spaCy index."""
        for token in self.tokens:
            if token.spacy_idx == spacy_idx:
                return token
        return None

    # Aggregate statistics
    @property
    def mean_surprisal(self) -> float:
        """Mean surprisal across all tokens."""
        if not self.tokens:
            return float("inf")
        return np.mean([t.mean_loss for t in self.tokens if np.isfinite(t.mean_loss)])

    @property
    def mean_loss(self) -> float:
        """Mean loss across all tokens."""
        if not self.tokens:
            return float("inf")
        return np.mean([t.mean_loss for t in self.tokens if np.isfinite(t.mean_loss)])

    @property
    def mean_entropy(self) -> float:
        """Mean entropy across all tokens."""
        if not self.tokens:
            return 0.0
        return np.mean(
            [t.mean_entropy for t in self.tokens if np.isfinite(t.mean_entropy)]
        )


class Predictor:
    """
    Unified class for calculating word predictability using either:
    - Masked Language Models (BERT-style): bidirectional context
    - Causal Language Models (GPT-style): left-to-right prediction
    """

    def __init__(
        self, tokenizer, model, model_type="masked", batch_size=32, device="cuda"
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            model: HuggingFace model (masked LM or causal LM)
            model_type: 'masked' or 'causal'
            batch_size: number of sequences to process in a batch
            device: 'cuda' or 'cpu'
        """
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.model.eval()

        self.model_type = model_type.lower()
        if self.model_type not in ["masked", "causal"]:
            raise ValueError("model_type must be 'masked' or 'causal'")

        # Get mask token ID for masked models
        if self.model_type == "masked":
            self.mask_id = tokenizer.mask_token_id
            if self.mask_id is None:
                raise ValueError("Tokenizer must have mask_token for masked model")

    def __call__(
        self, doc: Doc, window_size: Optional[int] = None
    ) -> DocPredictability:
        """
        Main entry point.

        Args:
            doc: spaCy Doc object
            window_size: Maximum context window (None for unlimited)
        """
        # Get token alignment
        token_map, trf_tok_ids = self.get_token_alignment(doc)

        # Determine effective window size
        seq_len = len(trf_tok_ids)
        if window_size is None or window_size >= seq_len:
            effective_window = seq_len
        else:
            effective_window = window_size

        if self.model_type == "masked":
            res = self._process_masked(doc, token_map, trf_tok_ids, effective_window)
        else:
            res = self._process_causal(doc, token_map, trf_tok_ids, effective_window)

        return res

    def get_token_alignment(
        self, doc: Doc
    ) -> tuple[dict[int, tuple[int, int]], npt.NDArray]:
        """
        Align spaCy tokens with transformer subword tokens using span overlap.
        Returns mapping and token IDs.
        """
        # Tokenize the full document
        encoding = self.tokenizer(
            doc.text,
            add_special_tokens=False,
            return_tensors="np",
            return_offsets_mapping=True,
        )
        trf_tok_ids = encoding.input_ids[0]
        offset_mapping = encoding["offset_mapping"][0]

        # Adjust offsets
        # GPT/Llama tokenizers start tokens on the whitespace before the word
        # So we need to adjust the start positions accordingly
        adjusted_starts = []

        for start, end in offset_mapping:
            token_text = doc.text[start:end]
            # Skip whitespace-only tokens
            # Find first non-whitespace character in token
            stripped_start = start + len(token_text) - len(token_text.lstrip())
            adjusted_starts.append(stripped_start)

        trf_starts = np.array(adjusted_starts)

        # Get index of the first character of each spaCy token
        spacy_starts = np.array([t.idx for t in doc])

        # Find the matching character indices (idx) --> 'common'
        # This is where both tokenizers align (both start new token at idx)
        # Get the index (i) of common in the spacy list --> 'common2spacy'
        # Get the index (i) of common in the trf token list --> 'common2trf'
        _, common2spacy, common2trf = np.intersect1d(
            spacy_starts, trf_starts, return_indices=True
        )

        # Build token map: spacy_idx -> (start_subword_idx, end_subword_idx)
        token_map = {}
        for spacy_ind, (st, end) in zip(common2spacy, pairwise(common2trf)):
            token_map[spacy_ind] = (st, end)

        return token_map, trf_tok_ids

    def _process_masked(
        self,
        doc: Doc,
        token_map: dict[int, tuple[int, int]],
        trf_tok_ids: npt.NDArray,
        window_size: int,
    ) -> DocPredictability:
        """
        Process with masked language model.
        Creates one windowed, masked sequence per token and batches them.
        """
        seq_len = len(trf_tok_ids)
        sequences_to_process = []
        spacy_indices = []
        mask_positions_list = []
        tok_indices_list = []

        # Create all masked sequences
        for spacy_idx, (subword_start, subword_end) in token_map.items():
            # Get centered window around this token
            win_start, win_end = get_centered_window(
                seq_len, subword_start, window_size
            )

            # Create masked version of this window
            window_ids = trf_tok_ids[win_start:win_end].copy()

            # Adjust token positions relative to window
            tok_start_in_window = subword_start - win_start
            tok_end_in_window = subword_end - win_start
            tok_indices = np.arange(tok_start_in_window, tok_end_in_window)

            # Mask the token
            window_ids[tok_indices] = self.mask_id

            sequences_to_process.append(window_ids)
            spacy_indices.append(spacy_idx)
            # Store where masks will be after adding special tokens (+1 for [CLS])
            mask_positions_list.append(tok_indices + 1)
            tok_indices_list.append((subword_start, subword_end))

        # Process in batches
        doc_predictability = DocPredictability(
            model_type=self.model_type, window_size=window_size
        )
        for i in range(0, len(sequences_to_process), self.batch_size):
            batch_seqs = sequences_to_process[i : i + self.batch_size]
            batch_spacy = spacy_indices[i : i + self.batch_size]
            batch_mask_pos = mask_positions_list[i : i + self.batch_size]
            batch_tok_inds = tok_indices_list[i : i + self.batch_size]

            # Prepare inputs
            inputs = self.tokenizer(
                self.tokenizer.batch_decode(batch_seqs),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Extract results for each token in batch
            for j, (spacy_idx, mask_pos, (sub_start, sub_end)) in enumerate(
                zip(batch_spacy, batch_mask_pos, batch_tok_inds)
            ):
                # Get actual token IDs
                actual_ids = trf_tok_ids[sub_start:sub_end]

                # Calculate metrics
                losses = []
                entropies = []

                for pos, actual_id in zip(mask_pos, actual_ids):
                    pred_logits = logits[j, pos]
                    probs_dist = torch.softmax(pred_logits, dim=-1)

                    # Calculate cross-entropy loss
                    target = torch.tensor([actual_id]).to(self.device)
                    loss = crossloss(pred_logits.unsqueeze(0), target).item()
                    losses.append(loss)

                    # Calculate entropy of the predicted distribution
                    # H(p) = -sum(p * log(p))
                    entropy = -torch.sum(
                        probs_dist * torch.log(probs_dist + 1e-10)
                    ).item()
                    entropies.append(entropy)

                # Store results
                token_pred = TokenPredictability(
                    spacy_idx=spacy_idx,
                    text=doc[spacy_idx].text,
                    subword_losses=losses,
                    subword_entropy=entropies,
                )
                doc_predictability.add_token(token_pred)

        return doc_predictability

    def _process_causal(
        self,
        doc: Doc,
        token_map: dict[int, tuple[int, int]],
        trf_tok_ids: npt.NDArray,
        window_size: int,
    ) -> DocPredictability:
        """
        Process with causal language model.

        For tokens within the first window_size positions: single forward pass.
        For tokens beyond: batch process with sliding windows.
        """
        doc_predictability = DocPredictability(
            model_type=self.model_type, window_size=window_size
        )
        seq_len = len(trf_tok_ids)

        # PHASE 1: Process first window_size tokens in one pass
        first_window_end = min(window_size, seq_len)
        first_window_ids = trf_tok_ids[:first_window_end]

        # Prepare input
        # Easiest to just convert to text and re-tokenize
        # HF is inconsistent about adding BOS tokens. Let's manually do that.
        input_str = self.tokenizer.bos_token + self.tokenizer.decode(first_window_ids)
        inputs = self.tokenizer(
            input_str,
            truncation=True,
            add_special_tokens=False,  # manually handle this
            return_tensors="pt",
        ).to(self.device)
        input_ids = inputs["input_ids"]  # Shape: [1, seq_len]

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Extract predictions for all tokens in this window
        for spacy_idx, (subword_start, subword_end) in token_map.items():
            # Only process tokens in the first window
            if subword_start >= first_window_end:
                break

            losses = []
            entropies = []

            # Position in input_ids (+1 for BOS token)
            pos = subword_start + 1

            # Predict from previous position
            # e.g., logit at BOS token predicts first real token in sequence
            pred_logits = logits[pos - 1]
            actual_id = input_ids[0, pos].item()

            # Calculate loss
            target = torch.tensor([actual_id]).to(self.device)
            loss = crossloss(pred_logits.unsqueeze(0), target).item()
            losses.append(loss)

            # Calculate entropy
            probs_dist = torch.softmax(pred_logits, dim=-1)
            entropy = -torch.sum(probs_dist * torch.log(probs_dist + 1e-10)).item()
            entropies.append(entropy)

            # Store results
            token_pred = TokenPredictability(
                spacy_idx=spacy_idx,
                text=doc[spacy_idx].text,
                subword_losses=losses,
                subword_entropy=entropies,
            )
            doc_predictability.add_token(token_pred)

        # PHASE 2: Process remaining tokens with sliding windows (batched)
        remaining_tokens = [
            (spacy_idx, sub_start, sub_end)
            for spacy_idx, (sub_start, sub_end) in token_map.items()
            if sub_start >= first_window_end
        ]

        if not remaining_tokens:
            return doc_predictability

        # Create windowed sequences for remaining tokens
        sequences_to_process = []
        spacy_indices = []

        for spacy_idx, subword_start, subword_end in remaining_tokens:
            # For causal model, window is all left context up to window_size
            win_start = max(0, subword_start - window_size + 1)
            win_end = subword_end

            window_ids = trf_tok_ids[win_start:win_end]
            sequences_to_process.append(window_ids)
            spacy_indices.append(spacy_idx)

        # Process in batches
        for i in range(0, len(sequences_to_process), self.batch_size):
            batch_seqs = sequences_to_process[i : i + self.batch_size]
            batch_spacy = spacy_indices[i : i + self.batch_size]

            # Prepare inputs
            input_strs = [
                self.tokenizer.bos_token + self.tokenizer.decode(seq)
                for seq in batch_seqs
            ]
            inputs = self.tokenizer(
                input_strs,
                add_special_tokens=False,  # manually handle this
                return_tensors="pt",
            ).to(self.device)
            input_ids = inputs["input_ids"]

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Extract results for each token in batch
            for j, spacy_idx in enumerate(batch_spacy):
                actual_id = trf_tok_ids[token_map[spacy_idx][0]]

                losses = []
                entropies = []

                # Predict from previous position
                pred_logits = logits[j, -1]
                probs_dist = torch.softmax(pred_logits, dim=-1)

                # Calculate loss
                target = torch.tensor([actual_id]).to(self.device)
                loss = crossloss(pred_logits.unsqueeze(0), target).item()
                losses.append(loss)

                # Calculate entropy
                entropy = -torch.sum(probs_dist * torch.log(probs_dist + 1e-10)).item()
                entropies.append(entropy)

                # Store results
                token_pred = TokenPredictability(
                    spacy_idx=spacy_idx,
                    text=doc[spacy_idx].text,
                    subword_losses=losses,
                    subword_entropy=entropies,
                )
                doc_predictability.add_token(token_pred)

        return doc_predictability

if __name__ == "__main__":
    import spacy
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
    
    # Load spaCy for tokenization
    nlp = spacy.load("en_core_web_sm")
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog."
    doc = nlp(text)
    
    print("=" * 60)
    print("Testing Masked LM (BERT)")
    print("=" * 60)
    
    # Test with masked LM
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    
    masked_predictor = Predictor(
        bert_tokenizer, 
        bert_model, 
        model_type="masked",
        batch_size=8,
        device="cpu"
    )
    
    masked_results = masked_predictor(doc, window_size=50)
    
    print(f"Mean surprisal: {masked_results.mean_surprisal:.4f}")
    print(f"Mean entropy: {masked_results.mean_entropy:.4f}")
    print("\nPer-token results:")
    for token in masked_results:
        print(f"  {token.text:12s} - prob: {token.mean_prob:.4f}, loss: {token.mean_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Testing Causal LM (GPT-2)")
    print("=" * 60)
    
    # Test with causal LM
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    causal_predictor = Predictor(
        gpt_tokenizer,
        gpt_model,
        model_type="causal",
        batch_size=8,
        device="cpu"
    )
    
    causal_results = causal_predictor(doc, window_size=4)
    
    print(f"Mean surprisal: {causal_results.mean_surprisal:.4f}")
    print(f"Mean entropy: {causal_results.mean_entropy:.4f}")
    print("\nPer-token results:")
    for token in causal_results:
        print(f"  {token.text:12s} - prob: {token.mean_prob:.4f}, loss: {token.mean_loss:.4f}")
        print(f"  {token.text:12s} - prob: {token.mean_prob:.4f}, loss: {token.mean_loss:.4f}")