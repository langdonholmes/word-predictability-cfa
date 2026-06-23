"""Compute n-gram and dependency bigram frequency tables from Dolma DocBins.

Counts 3-grams and dependency bigrams across a corpus's DocBins and writes
parquet frequency tables.

To stay within RAM on a ~1B-token corpus (a single in-memory Counter does
not fit), counting is *sharded*: each ``.spacy`` file is counted on its own
into a partial parquet (no ``min_count`` filter), then all partials are merged
with polars' streaming engine, summing counts per key out-of-core and applying
``min_count`` only after the global sum. The per-shard step is restartable —
shards whose partials already exist are skipped.

Usage:
    python src/pipeline/4_collate_dolma.py --corpus a|b \
        [--min-count 2] [--keep-partials]
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from util.collation import (
    count_ngrams_and_depgrams,
    counter_to_depgram_parquet,
    counter_to_ngram_parquet,
    merge_partial_parquets,
)
from util.paths import DOLMA_DIR, DOLMA_DOCBINS_DIR, DOLMA_FREQ_DIR
from util.process_docs import load_docbin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N = 3  # trigrams

NGRAM_KEYS = [f"token_{i}" for i in range(N)]
DEPGRAM_KEYS = [
    "head_lemma",
    "head_tag",
    "relation",
    "dependent_lemma",
    "dependent_tag",
]


def count_shards(docbin_dir: Path, partial_dir: Path) -> None:
    """Count each DocBin file into per-shard partial parquets.

    Restartable: a shard whose ``.done`` marker exists is skipped. Partials
    are written with ``min_count=1`` (no filtering) so the global merge can
    apply the real threshold after summing across shards.
    """
    partial_dir.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(docbin_dir.glob("*.spacy"))
    logger.info("Counting %d shards from %s", len(shard_files), docbin_dir)

    for idx, shard in enumerate(shard_files):
        done = partial_dir / f"{shard.stem}.done"
        if done.exists():
            logger.info("[%d/%d] %s — already counted, skipping",
                        idx + 1, len(shard_files), shard.name)
            continue

        logger.info("[%d/%d] counting %s …", idx + 1, len(shard_files), shard.name)
        ngram_counter, dep_counter = count_ngrams_and_depgrams(
            load_docbin(shard), n=N,
        )

        # Drop any stale/partial output for this shard, then write fresh.
        ng_path = partial_dir / f"{shard.stem}.ngrams.parquet"
        dep_path = partial_dir / f"{shard.stem}.depgrams.parquet"
        ng_df = counter_to_ngram_parquet(ngram_counter, n=N, min_count=1)
        ng_df.write_parquet(ng_path, compression="zstd")
        dep_df = counter_to_depgram_parquet(dep_counter, min_count=1)
        dep_df.write_parquet(dep_path, compression="zstd")

        # Marker written last so an interrupted shard is recomputed next run.
        done.write_text(json.dumps({
            "ngram_rows": len(ng_df),
            "depgram_rows": len(dep_df),
        }))
        logger.info(
            "[%d/%d] %s done — %s n-grams, %s depgrams",
            idx + 1, len(shard_files), shard.name,
            f"{len(ng_df):,}", f"{len(dep_df):,}",
        )


def collate_corpus(
    corpus: str,
    min_count: int = 2,
    keep_partials: bool = False,
):
    docbin_dir = DOLMA_DOCBINS_DIR / f"corpus_{corpus}"
    if not docbin_dir.exists() or not any(docbin_dir.glob("*.spacy")):
        raise FileNotFoundError(
            f"No DocBins found at {docbin_dir}. Run 3_spacy_dolma.py first."
        )

    out_dir = DOLMA_FREQ_DIR / f"corpus_{corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / "partials"

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

    # --- Stage 1: per-shard counting (memory bounded, restartable) ---
    count_shards(docbin_dir, partial_dir)

    # --- Stage 2: streaming merge of partials ---
    ngram_path = out_dir / "3grams.parquet"
    dep_path = out_dir / "depgrams.parquet"

    logger.info("Merging n-gram partials (min_count=%d) …", min_count)
    n_ngrams = merge_partial_parquets(
        str(partial_dir / "*.ngrams.parquet"), NGRAM_KEYS, ngram_path, min_count,
    )
    logger.info("Wrote %s (%s rows)", ngram_path, f"{n_ngrams:,}")

    logger.info("Merging depgram partials (min_count=%d) …", min_count)
    n_dep = merge_partial_parquets(
        str(partial_dir / "*.depgrams.parquet"), DEPGRAM_KEYS, dep_path, min_count,
    )
    logger.info("Wrote %s (%s rows)", dep_path, f"{n_dep:,}")

    # --- Verification (lazy/streaming so we never load the full table) ---
    verify(corpus, ngram_path, dep_path, min_count, expected_tokens)

    if not keep_partials:
        logger.info("Removing partials at %s", partial_dir)
        shutil.rmtree(partial_dir, ignore_errors=True)

    logger.info("Done — output in %s", out_dir)


def verify(corpus, ngram_path, dep_path, min_count, expected_tokens):
    logger.info("--- Verification ---")

    ng = pl.scan_parquet(ngram_path)
    dep = pl.scan_parquet(dep_path)

    total_3gram = ng.select(pl.col("count").sum()).collect().item()
    n_ngrams = ng.select(pl.len()).collect().item()
    n_dep = dep.select(pl.len()).collect().item()
    logger.info("3-gram count sum: %s (includes padding)", f"{total_3gram:,}")
    logger.info("Unique 3-grams (count >= %d): %s", min_count, f"{n_ngrams:,}")
    logger.info("Unique depgrams (count >= %d): %s", min_count, f"{n_dep:,}")

    logger.info("Top-20 3-grams:")
    top_ng = ng.top_k(20, by="count").collect()
    for row in top_ng.iter_rows(named=True):
        gram = tuple(row[f"token_{i}"] for i in range(N))
        logger.info("  %s  %s", f"{row['count']:>12,}", gram)

    logger.info("Top-20 depgrams:")
    top_dep = dep.top_k(20, by="count").collect()
    for row in top_dep.iter_rows(named=True):
        logger.info(
            "  %s  %s -%s-> %s",
            f"{row['count']:>12,}",
            f"{row['head_lemma']}/{row['head_tag']}",
            row["relation"],
            f"{row['dependent_lemma']}/{row['dependent_tag']}",
        )

    # Unigram check: sum counts grouped by the trailing token.
    unigram = (
        ng.group_by("token_2")
        .agg(pl.col("count").sum().alias("freq"))
        .sort("freq", descending=True)
        .collect(engine="streaming")
    )
    unigram_total = unigram.select(pl.col("freq").sum()).item()
    logger.info("Unigram total (from token_2 groupby): %s", f"{unigram_total:,}")
    if expected_tokens:
        ratio = unigram_total / expected_tokens
        logger.info(
            "Ratio to expected whitespace tokens: %.4f "
            "(>1.0 expected due to padding, <1.0 if min_count filtered heavily)",
            ratio,
        )
    logger.info("Top-20 unigrams:")
    for row in unigram.head(20).iter_rows(named=True):
        logger.info("  %s  %s", f"{row['freq']:>12,}", row["token_2"])


def main():
    parser = argparse.ArgumentParser(
        description="Compute frequency tables from Dolma DocBins"
    )
    parser.add_argument("--corpus", required=True, choices=["a", "b"])
    parser.add_argument(
        "--min-count", type=int, default=2,
        help="Minimum global count to include in parquet (default: 2)",
    )
    parser.add_argument(
        "--keep-partials", action="store_true",
        help="Keep per-shard partial parquets after merging (default: delete)",
    )
    args = parser.parse_args()

    collate_corpus(
        corpus=args.corpus,
        min_count=args.min_count,
        keep_partials=args.keep_partials,
    )


if __name__ == "__main__":
    main()
