"""Collate n-gram and dependency-gram counts from SlimPajama DocBins."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter

import polars as pl
from tqdm.auto import tqdm

from util.paths import DATA_DIR
from util.process_docs import load_all_docbins


def get_ngrams(docs, output_dir=None, n=3) -> None:
    """Count n-grams in a collection of spaCy docs."""
    ngram_counter = Counter()

    for doc in tqdm(docs, desc="Processing N-grams"):
        tokens = ["#"] * (n - 1) + [token.lower_ for token in doc if token.is_alpha]
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngram_counter.update(ngrams)

    data = {f"token_{i}": [] for i in range(n)}
    data["count"] = []

    for ngram, count in ngram_counter.items():
        for i, token in enumerate(ngram):
            data[f"token_{i}"].append(token)
        data["count"].append(count)

    ngram_counts = pl.DataFrame(data).sort("count", descending=True)
    print(ngram_counts)

    if output_dir is not None:
        ngram_counts.write_parquet(output_dir / f"{n}grams.parquet", compression="zstd")


def get_depgrams(docs, output_dir=None) -> None:
    """Count dependency bigrams in a collection of spaCy docs."""
    depgram_counter = Counter()

    for doc in tqdm(docs, desc="Processing Dependency Grams"):
        depgrams = [
            (tok.head.lemma_, tok.head.tag_, tok.dep_, tok.lemma_, tok.tag_)
            for tok in doc
            if tok.is_alpha and tok.head.is_alpha
        ]
        depgram_counter.update(depgrams)

    ngram_counts = pl.DataFrame(
        [
            (head, head_tag, relation, dep, dep_tag, count)
            for (head, head_tag, relation, dep, dep_tag), count in depgram_counter.items()
        ],
        schema=[
            "head_lemma",
            "head_tag",
            "relation",
            "dependent_lemma",
            "dependent_tag",
            "count",
        ],
        orient="row",
    ).sort("count", descending=True)

    print(ngram_counts)

    if output_dir is not None:
        ngram_counts.write_parquet(
            output_dir / "depgrams.parquet", compression="zstd"
        )


def main():
    output_dir = DATA_DIR / "slim_pajama_lists"
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = list(
        tqdm(
            load_all_docbins(str(DATA_DIR / "slim_pajama_docbins")),
            desc="Loading Docs",
            total=458_047,
        )
    )

    get_ngrams(docs, output_dir=output_dir, n=3)
    get_depgrams(docs, output_dir=output_dir)


if __name__ == "__main__":
    main()
