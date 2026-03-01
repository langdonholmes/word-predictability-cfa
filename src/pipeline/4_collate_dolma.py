"""Compute n-gram and dependency bigram frequency tables from Dolma DocBins.

Streams through all DocBin files for a corpus, counts 3-grams and depgrams
in a single pass, and writes parquet frequency tables.

Usage:
    python src/pipeline/4_collate_dolma.py --corpus a|b \
        [--min-count 2] [--singleton-prune 0]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.collation import (
    count_ngrams_and_depgrams,
    counter_to_depgram_parquet,
    counter_to_ngram_parquet,
)
from util.paths import DOLMA_DIR, DOLMA_DOCBINS_DIR, DOLMA_FREQ_DIR
from util.process_docs import load_all_docbins

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N = 3  # trigrams


def collate_corpus(
    corpus: str,
    min_count: int = 2,
    singleton_prune: int = 0,
):
    docbin_dir = DOLMA_DOCBINS_DIR / f"corpus_{corpus}"
    if not docbin_dir.exists() or not any(docbin_dir.glob("*.spacy")):
        raise FileNotFoundError(
            f"No DocBins found at {docbin_dir}. Run 3_spacy_dolma.py first."
        )

    out_dir = DOLMA_FREQ_DIR / f"corpus_{corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load expected stats for verification
    summary_path = DOLMA_DIR / "preprocessed" / f"corpus_{corpus}_preprocess_summary.json"
    expected_tokens = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        expected_tokens = summary.get("total_tokens")
        logger.info(
            "Corpus %s: expecting ~%s whitespace tokens",
            corpus, f"{expected_tokens:,}" if expected_tokens else "unknown",
        )

    # Single-pass counting
    logger.info("Streaming DocBins from %s …", docbin_dir)
    docs = load_all_docbins(docbin_dir)
    ngram_counter, dep_counter = count_ngrams_and_depgrams(
        docs, n=N, singleton_prune_interval=singleton_prune,
    )

    # Convert to DataFrames
    logger.info("Building parquet tables (min_count=%d) …", min_count)
    ngram_df = counter_to_ngram_parquet(ngram_counter, n=N, min_count=min_count)
    dep_df = counter_to_depgram_parquet(dep_counter, min_count=min_count)

    # Write parquet with zstd compression
    ngram_path = out_dir / "3grams.parquet"
    dep_path = out_dir / "depgrams.parquet"

    ngram_df.write_parquet(ngram_path, compression="zstd")
    logger.info("Wrote %s (%s rows)", ngram_path, f"{len(ngram_df):,}")

    dep_df.write_parquet(dep_path, compression="zstd")
    logger.info("Wrote %s (%s rows)", dep_path, f"{len(dep_df):,}")

    # --- Verification ---
    logger.info("--- Verification ---")

    # Unigram sum from 3-gram table (group by last token position)
    unigram_sum = ngram_df.select("count").sum().item()
    logger.info(
        "3-gram count sum: %s (includes padding + filtered entries)",
        f"{unigram_sum:,}",
    )

    # Unique entry counts
    logger.info("Unique 3-grams (count >= %d): %s", min_count, f"{len(ngram_df):,}")
    logger.info("Unique depgrams (count >= %d): %s", min_count, f"{len(dep_df):,}")

    # Top-20 3-grams
    logger.info("Top-20 3-grams:")
    for row in ngram_df.head(20).iter_rows(named=True):
        gram = (row["token_0"], row["token_1"], row["token_2"])
        logger.info("  %s  %s", f"{row['count']:>12,}", gram)

    # Top-20 depgrams
    logger.info("Top-20 depgrams:")
    for row in dep_df.head(20).iter_rows(named=True):
        logger.info(
            "  %s  %s -%s-> %s",
            f"{row['count']:>12,}",
            f"{row['head_lemma']}/{row['head_tag']}",
            row["relation"],
            f"{row['dependent_lemma']}/{row['dependent_tag']}",
        )

    # Unigram frequency check: group by token_2 and sum
    unigram_df = ngram_df.group_by("token_2").agg(
        ngram_df["count"].sum().alias("freq")
    ).sort("freq", descending=True)
    unigram_total = unigram_df.select("freq").sum().item()
    logger.info(
        "Unigram total (from token_2 groupby): %s", f"{unigram_total:,}",
    )
    if expected_tokens:
        ratio = unigram_total / expected_tokens
        logger.info(
            "Ratio to expected whitespace tokens: %.4f "
            "(>1.0 expected due to padding, <1.0 if min_count filtered heavily)",
            ratio,
        )

    logger.info("Top-20 unigrams:")
    for row in unigram_df.head(20).iter_rows(named=True):
        logger.info("  %s  %s", f"{row['freq']:>12,}", row["token_2"])

    logger.info("Done — output in %s", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compute frequency tables from Dolma DocBins"
    )
    parser.add_argument("--corpus", required=True, choices=["a", "b"])
    parser.add_argument(
        "--min-count", type=int, default=2,
        help="Minimum count to include in parquet (default: 2)",
    )
    parser.add_argument(
        "--singleton-prune", type=int, default=0,
        help="Prune singletons every N docs during counting (0=disabled)",
    )
    args = parser.parse_args()

    collate_corpus(
        corpus=args.corpus,
        min_count=args.min_count,
        singleton_prune=args.singleton_prune,
    )


if __name__ == "__main__":
    main()
