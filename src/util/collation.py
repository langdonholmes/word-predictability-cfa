"""Shared frequency-counting logic for n-grams and dependency bigrams.

Provides single-pass streaming counters and parquet serialization.
"""

import logging
from collections import Counter
from typing import Iterable, Optional

import polars as pl
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

PAD = "#"


def count_ngrams_and_depgrams(
    docs_iter: Iterable[Doc],
    n: int = 3,
    total: Optional[int] = None,
    singleton_prune_interval: int = 0,
) -> tuple[Counter, Counter]:
    """Single-pass streaming count of n-grams and dependency bigrams.

    Parameters
    ----------
    docs_iter : iterable of spacy.tokens.Doc
        Streaming doc iterator (e.g. from load_all_docbins).
    n : int
        N-gram order (default 3).
    total : int or None
        Expected total docs, for logging progress.
    singleton_prune_interval : int
        If > 0, prune singleton entries from counters every this many docs
        as a memory safety valve. 0 disables pruning.

    Returns
    -------
    (ngram_counter, depgram_counter) : tuple[Counter, Counter]
        ngram keys   : tuple of n lowercased token strings
        depgram keys : (head_lemma, head_tag, relation, dep_lemma, dep_tag)
    """
    ngram_counter: Counter = Counter()
    dep_counter: Counter = Counter()
    doc_count = 0

    pad_prefix = (PAD,) * (n - 1)

    for doc in docs_iter:
        # --- n-grams (with boundary padding) ---
        tokens = pad_prefix + tuple(t.text.lower() for t in doc)
        for i in range(len(tokens) - n + 1):
            ngram_counter[tokens[i : i + n]] += 1

        # --- dependency bigrams ---
        for token in doc:
            if token.dep_ == "ROOT" or token.dep_ == "punct":
                continue
            key = (
                token.head.lemma_.lower(),
                token.head.tag_,
                token.dep_,
                token.lemma_.lower(),
                token.tag_,
            )
            dep_counter[key] += 1

        doc_count += 1

        if doc_count % 500_000 == 0:
            logger.info(
                "  %s docs counted  |  %s unique n-grams  |  %s unique depgrams",
                f"{doc_count:,}",
                f"{len(ngram_counter):,}",
                f"{len(dep_counter):,}",
            )

        # Memory safety valve: prune singletons periodically
        if (
            singleton_prune_interval > 0
            and doc_count % singleton_prune_interval == 0
        ):
            before_ng = len(ngram_counter)
            before_dep = len(dep_counter)
            ngram_counter = Counter(
                {k: v for k, v in ngram_counter.items() if v > 1}
            )
            dep_counter = Counter(
                {k: v for k, v in dep_counter.items() if v > 1}
            )
            logger.info(
                "  Singleton prune @ %s docs: n-grams %s→%s, depgrams %s→%s",
                f"{doc_count:,}",
                f"{before_ng:,}",
                f"{len(ngram_counter):,}",
                f"{before_dep:,}",
                f"{len(dep_counter):,}",
            )

    logger.info(
        "Counting complete: %s docs, %s unique n-grams, %s unique depgrams",
        f"{doc_count:,}",
        f"{len(ngram_counter):,}",
        f"{len(dep_counter):,}",
    )
    return ngram_counter, dep_counter


def counter_to_ngram_parquet(
    counter: Counter, n: int, min_count: int = 2
) -> pl.DataFrame:
    """Convert an n-gram Counter to a sorted Polars DataFrame.

    Parameters
    ----------
    counter : Counter
        Keys are tuples of n strings.
    n : int
        N-gram order (determines column names token_0 … token_{n-1}).
    min_count : int
        Drop entries with count < min_count.

    Returns
    -------
    pl.DataFrame with columns token_0, ..., token_{n-1}, count
    """
    cols = {f"token_{i}": [] for i in range(n)}
    counts = []

    for gram, cnt in counter.items():
        if cnt < min_count:
            continue
        for i in range(n):
            cols[f"token_{i}"].append(gram[i])
        counts.append(cnt)

    df = pl.DataFrame({**cols, "count": counts})
    df = df.sort("count", descending=True)

    logger.info(
        "N-gram table: %s rows (min_count=%d filtered from %s unique)",
        f"{len(df):,}",
        min_count,
        f"{len(counter):,}",
    )
    return df


def counter_to_depgram_parquet(
    counter: Counter, min_count: int = 2
) -> pl.DataFrame:
    """Convert a dependency-bigram Counter to a sorted Polars DataFrame.

    Parameters
    ----------
    counter : Counter
        Keys are (head_lemma, head_tag, relation, dep_lemma, dep_tag).
    min_count : int
        Drop entries with count < min_count.

    Returns
    -------
    pl.DataFrame with columns head_lemma, head_tag, relation,
        dependent_lemma, dependent_tag, count
    """
    rows = {
        "head_lemma": [],
        "head_tag": [],
        "relation": [],
        "dependent_lemma": [],
        "dependent_tag": [],
        "count": [],
    }

    for (h_lem, h_tag, rel, d_lem, d_tag), cnt in counter.items():
        if cnt < min_count:
            continue
        rows["head_lemma"].append(h_lem)
        rows["head_tag"].append(h_tag)
        rows["relation"].append(rel)
        rows["dependent_lemma"].append(d_lem)
        rows["dependent_tag"].append(d_tag)
        rows["count"].append(cnt)

    df = pl.DataFrame(rows)
    df = df.sort("count", descending=True)

    logger.info(
        "Depgram table: %s rows (min_count=%d filtered from %s unique)",
        f"{len(df):,}",
        min_count,
        f"{len(counter):,}",
    )
    return df
