import math
import numpy as np
from lexicalrichness import LexicalRichness


def mtld(doc) -> float:
    """Measure of Textual Lexical Diversity (threshold=0.72)."""
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha]
    if len(tokens) < 10:
        return np.nan
    token_str = " ".join(tokens)
    lex = LexicalRichness(token_str)
    return lex.mtld(threshold=0.72)


def lexical_density(doc) -> float:
    """Proportion of content-word tokens (NOUN, VERB, ADJ, ADV) among alpha tokens."""
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    words = [t for t in doc if t.is_alpha]
    if not words:
        return np.nan
    content_words = [t for t in words if t.pos_ in content_pos]
    return len(content_words) / len(words)


def log_mean_token_freq(doc, token_freq: dict, total_tokens: int) -> float:
    """Mean log10 relative frequency per alpha token."""
    log_freqs = []
    for token in doc:
        if token.is_alpha:
            freq = token_freq.get(token.lemma_.lower(), 1)
            log_freqs.append(math.log10(freq / total_tokens))
    return np.mean(log_freqs) if log_freqs else np.nan


def mean_word_length(doc) -> float:
    """Mean number of characters per alpha token."""
    lengths = [len(t.text) for t in doc if t.is_alpha]
    return np.mean(lengths) if lengths else np.nan
