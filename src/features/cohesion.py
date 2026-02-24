import numpy as np

CONNECTIVES = frozenset(
    {
        "also",
        "although",
        "and",
        "because",
        "besides",
        "but",
        "consequently",
        "conversely",
        "finally",
        "first",
        "firstly",
        "for",
        "furthermore",
        "hence",
        "however",
        "if",
        "indeed",
        "instead",
        "lastly",
        "likewise",
        "meanwhile",
        "moreover",
        "nevertheless",
        "next",
        "nonetheless",
        "nor",
        "once",
        "or",
        "otherwise",
        "overall",
        "second",
        "secondly",
        "similarly",
        "since",
        "so",
        "still",
        "subsequently",
        "then",
        "therefore",
        "third",
        "thirdly",
        "though",
        "thus",
        "unless",
        "whereas",
        "while",
        "yet",
    }
)


def content_word_overlap(doc) -> float:
    """Mean Jaccard overlap of content-word lemma sets between adjacent sentences."""
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    sents = list(doc.sents)
    if len(sents) < 2:
        return np.nan

    lemma_sets = []
    for sent in sents:
        lemmas = {t.lemma_.lower() for t in sent if t.pos_ in content_pos}
        lemma_sets.append(lemmas)

    overlaps = []
    for i in range(len(lemma_sets) - 1):
        a, b = lemma_sets[i], lemma_sets[i + 1]
        union = a | b
        if not union:
            overlaps.append(0.0)
        else:
            overlaps.append(len(a & b) / len(union))

    return np.mean(overlaps)


def connective_density(doc) -> float:
    """Connective tokens per sentence (Coh-Metrix style)."""
    sents = list(doc.sents)
    if not sents:
        return np.nan
    total = sum(1 for t in doc if t.text.lower() in CONNECTIVES)
    return total / len(sents)


def sentence_similarity(doc) -> float:
    """Mean cosine similarity of adjacent sentence vectors (GloVe 300-dim)."""
    sents = list(doc.sents)
    if len(sents) < 2:
        return np.nan

    sims = []
    for i in range(len(sents) - 1):
        v1 = sents[i].vector
        v2 = sents[i + 1].vector
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
        sims.append(np.dot(v1, v2) / (norm1 * norm2))

    return np.mean(sims) if sims else np.nan
