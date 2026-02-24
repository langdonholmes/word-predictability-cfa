import numpy as np


def words_per_sentence(doc) -> float:
    """Mean non-punctuation tokens per sentence."""
    counts = []
    for sent in doc.sents:
        n = sum(1 for t in sent if not t.is_punct)
        counts.append(n)
    return np.mean(counts) if counts else np.nan


def clauses_per_tunit(doc) -> float:
    """Sentence count divided by ROOT count (T-units)."""
    sents = list(doc.sents)
    roots = sum(1 for t in doc if t.dep_ == "ROOT")
    if roots == 0:
        return np.nan
    return len(sents) / roots


def mod_per_nom(doc) -> float:
    """Modifier children (amod, det, nummod, compound) per NOUN token."""
    nominals = [t for t in doc if t.pos_ == "NOUN"]
    if not nominals:
        return np.nan
    total_mods = 0
    for nom in nominals:
        total_mods += sum(
            1 for c in nom.children if c.dep_ in {"amod", "det", "nummod", "compound"}
        )
    return total_mods / len(nominals)


def mean_dep_distance(doc) -> float:
    """Mean absolute distance between each token and its head (excluding ROOT)."""
    distances = [abs(t.i - t.head.i) for t in doc if t.dep_ != "ROOT"]
    return np.mean(distances) if distances else np.nan
