"""Process preprocessed Dolma JSONL through spaCy using manual parallelism.

Each worker reads the same gzipped JSONL and processes every Nth doc
(stride-based), avoiding the need to shard to disk. Per-doc processing
with error handling ensures one bad doc doesn't kill a worker.

Usage:
    python src/pipeline/3_spacy_dolma.py --corpus a|b \
        [--n-workers 8] [--tokens-per-file 10000000]
"""

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy
from spacy.tokens import DocBin
from tqdm.auto import tqdm

from util.paths import DOLMA_DIR, DOLMA_DOCBINS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_summary(corpus: str) -> dict:
    path = DOLMA_DIR / "preprocessed" / f"corpus_{corpus}_preprocess_summary.json"
    with open(path) as f:
        return json.load(f)


def texts(input_path: Path, worker_id: int, n_workers: int):
    """Yield (text, meta) for every Nth doc assigned to this worker."""
    with gzip.open(input_path, "rt", encoding="utf-8") as gz:
        for i, line in enumerate(gz):
            if i % n_workers != worker_id:
                continue
            record = json.loads(line)
            yield record["text"], {"dolma_id": record["dolma_id"], "source": record["source"]}


def process_worker(
    worker_id: int,
    input_path: Path,
    n_workers: int,
    output_dir: Path,
    tokens_per_file: int,
    counter: mp.Value,
    error_counter: mp.Value,
):
    """Load spaCy, process assigned docs single-threaded, write DocBins."""
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000

    current_docs = []
    current_tokens = 0
    docbin_idx = 0
    local_count = 0
    local_skipped = 0

    for text, meta in texts(input_path, worker_id, n_workers):
        try:
            doc = nlp(text)
        except Exception as e:
            logger.warning("Worker %d: skipped %s (%s): %s", worker_id, meta["dolma_id"], meta["source"], e)
            local_skipped += 1
            if local_skipped >= 10:
                with error_counter.get_lock():
                    error_counter.value += local_skipped
                local_skipped = 0
            continue

        doc.user_data["meta"] = meta
        current_docs.append(doc)
        current_tokens += len(doc)
        local_count += 1

        if current_tokens >= tokens_per_file:
            out = output_dir / f"shard{worker_id:02d}_{docbin_idx:04d}.spacy"
            DocBin(docs=current_docs, store_user_data=True).to_disk(out)
            docbin_idx += 1
            current_docs = []
            current_tokens = 0

        # Update shared counter in batches to reduce lock contention
        if local_count >= 100:
            with counter.get_lock():
                counter.value += local_count
            local_count = 0

    # Flush remaining docs
    if current_docs:
        out = output_dir / f"shard{worker_id:02d}_{docbin_idx:04d}.spacy"
        DocBin(docs=current_docs, store_user_data=True).to_disk(out)

    # Final counter updates
    if local_count:
        with counter.get_lock():
            counter.value += local_count
    if local_skipped:
        with error_counter.get_lock():
            error_counter.value += local_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Process Dolma corpus through spaCy (manual parallelism)"
    )
    parser.add_argument("--corpus", required=True, choices=["a", "b"])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--tokens-per-file", type=int, default=10_000_000)
    args = parser.parse_args()

    summary = load_summary(args.corpus)
    expected_docs = summary["total_docs"]
    logger.info(
        "Corpus %s: %s docs, ~%s tokens",
        args.corpus, f"{expected_docs:,}", f"{summary['total_tokens']:,}",
    )

    output_dir = DOLMA_DOCBINS_DIR / f"corpus_{args.corpus}"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = DOLMA_DIR / "preprocessed" / f"corpus_{args.corpus}.jsonl.gz"

    # Launch workers
    counter = mp.Value("i", 0)
    error_counter = mp.Value("i", 0)
    workers = []
    for i in range(args.n_workers):
        p = mp.Process(
            target=process_worker,
            args=(i, input_path, args.n_workers, output_dir, args.tokens_per_file, counter, error_counter),
        )
        p.start()
        workers.append(p)

    logger.info("Launched %d workers (stride-based, per-doc processing)", len(workers))

    # Monitor progress
    try:
        with tqdm(total=expected_docs, desc=f"spaCy corpus_{args.corpus}", unit=" docs") as pbar:
            while any(p.is_alive() for p in workers):
                pbar.update(counter.value - pbar.n)
                time.sleep(2)
            pbar.update(counter.value - pbar.n)
    except KeyboardInterrupt:
        logger.warning("Interrupted — terminating workers")
        for p in workers:
            p.terminate()
        for p in workers:
            p.join(timeout=10)
        sys.exit(1)

    # Check exit codes
    failed = [i for i, p in enumerate(workers) if p.exitcode != 0]
    if failed:
        logger.error(
            "Workers %s failed! Exit codes: %s",
            failed, [workers[i].exitcode for i in failed],
        )
        sys.exit(1)

    # Summary
    processed = counter.value
    skipped = error_counter.value
    logger.info(
        "Processed: %s, Skipped: %s, Total: %s (expected %s)",
        f"{processed:,}", f"{skipped:,}", f"{processed + skipped:,}", f"{expected_docs:,}",
    )
    if processed + skipped == expected_docs:
        logger.info("PASS: doc count matches")
    else:
        logger.warning("MISMATCH: delta %d", (processed + skipped) - expected_docs)


if __name__ == "__main__":
    main()
