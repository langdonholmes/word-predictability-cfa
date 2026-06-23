"""Process preprocessed Dolma JSONL through spaCy using manual parallelism.

Each worker reads the same gzipped JSONL and processes every Nth doc
(stride-based), avoiding the need to shard to disk. Per-doc processing
with error handling ensures one bad doc doesn't kill a worker.

The stage is idempotent at the worker level: a worker writes a
``shard<NN>.done`` sentinel only after finishing its full stride, so
re-running skips completed workers and reprocesses only those that never
finished (clearing their stale/partial shards first). Because the stride
assignment is ``i % n_workers``, resuming is only valid with the same
``--n-workers`` as the original run; a mismatch is rejected.

Usage:
    python src/reference_corpus/3_parse.py --corpus a|b \
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


def worker_sentinel(output_dir: Path, worker_id: int) -> Path:
    return output_dir / f"shard{worker_id:02d}.done"


def read_sentinel(output_dir: Path, worker_id: int) -> dict | None:
    """Return a worker's completion record, or None if absent/unreadable."""
    path = worker_sentinel(output_dir, worker_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


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
    """Load spaCy, process assigned docs single-threaded, write DocBins.

    Idempotent: clears any stale/partial shards for this worker before
    starting, and writes its ``.done`` sentinel only after a clean finish
    so an interrupted worker is reprocessed from scratch next time.
    """
    # Drop stale/partial output (incl. 0-byte stubs) from a prior failed run.
    for stale in output_dir.glob(f"shard{worker_id:02d}_*.spacy"):
        stale.unlink()

    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000

    current_docs = []
    current_tokens = 0
    docbin_idx = 0
    local_count = 0
    local_skipped = 0
    total_processed = 0
    total_skipped = 0

    for text, meta in texts(input_path, worker_id, n_workers):
        try:
            doc = nlp(text)
        except Exception as e:
            logger.warning("Worker %d: skipped %s (%s): %s", worker_id, meta["dolma_id"], meta["source"], e)
            local_skipped += 1
            total_skipped += 1
            if local_skipped >= 10:
                with error_counter.get_lock():
                    error_counter.value += local_skipped
                local_skipped = 0
            continue

        doc.user_data["meta"] = meta
        current_docs.append(doc)
        current_tokens += len(doc)
        local_count += 1
        total_processed += 1

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
        docbin_idx += 1

    # Final counter updates
    if local_count:
        with counter.get_lock():
            counter.value += local_count
    if local_skipped:
        with error_counter.get_lock():
            error_counter.value += local_skipped

    # Written last: a crash before this leaves no sentinel, so the worker
    # is treated as incomplete and reprocessed on the next run.
    with open(worker_sentinel(output_dir, worker_id), "w") as f:
        json.dump(
            {
                "n_workers": n_workers,
                "tokens_per_file": tokens_per_file,
                "processed": total_processed,
                "skipped": total_skipped,
                "docbins": docbin_idx,
            },
            f,
        )


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

    # Resume: which workers already finished their full stride?
    completed = {}
    for wid in range(args.n_workers):
        sentinel = read_sentinel(output_dir, wid)
        if sentinel is not None:
            completed[wid] = sentinel

    # Guard: stride mapping (i % n_workers) is only stable if n_workers is
    # unchanged. Resuming with a different count would parse the wrong docs.
    mismatched = sorted(
        {s.get("n_workers") for s in completed.values() if s.get("n_workers") != args.n_workers}
    )
    if mismatched:
        logger.error(
            "n_workers mismatch: existing sentinels used %s but --n-workers=%d "
            "was requested. Re-run with the original worker count, or clear "
            "%s to reprocess from scratch.",
            mismatched, args.n_workers, output_dir,
        )
        sys.exit(1)

    todo = [wid for wid in range(args.n_workers) if wid not in completed]
    if completed:
        logger.info(
            "Resuming: %d/%d workers already complete %s",
            len(completed), args.n_workers, sorted(completed),
        )
    if not todo:
        logger.info("All %d workers complete — nothing to do.", args.n_workers)
        return
    logger.info("Workers to run: %s", todo)

    # Launch only the incomplete workers
    counter = mp.Value("i", 0)
    error_counter = mp.Value("i", 0)
    workers = {}
    for i in todo:
        p = mp.Process(
            target=process_worker,
            args=(i, input_path, args.n_workers, output_dir, args.tokens_per_file, counter, error_counter),
        )
        p.start()
        workers[i] = p

    logger.info("Launched %d workers (stride-based, per-doc processing)", len(workers))

    # Docs already on disk from completed workers, for an accurate bar.
    done_processed = sum((s.get("processed") or 0) for s in completed.values())
    have_done_counts = all(s.get("processed") is not None for s in completed.values())

    # Monitor progress
    try:
        with tqdm(
            total=expected_docs, initial=done_processed,
            desc=f"spaCy corpus_{args.corpus}", unit=" docs",
        ) as pbar:
            while any(p.is_alive() for p in workers.values()):
                pbar.update(done_processed + counter.value - pbar.n)
                time.sleep(2)
            pbar.update(done_processed + counter.value - pbar.n)
    except KeyboardInterrupt:
        logger.warning("Interrupted — terminating workers")
        for p in workers.values():
            p.terminate()
        for p in workers.values():
            p.join(timeout=10)
        sys.exit(1)

    # Check exit codes
    failed = [i for i, p in workers.items() if p.exitcode != 0]
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
        "This run — processed: %s, skipped: %s across %d workers",
        f"{processed:,}", f"{skipped:,}", len(workers),
    )
    if have_done_counts:
        done_skipped = sum(s.get("skipped", 0) for s in completed.values())
        total = done_processed + done_skipped + processed + skipped
        logger.info("Total — processed + skipped: %s (expected %s)", f"{total:,}", f"{expected_docs:,}")
        if total == expected_docs:
            logger.info("PASS: doc count matches")
        else:
            logger.warning("MISMATCH: delta %d", total - expected_docs)
    elif completed:
        logger.info(
            "%d workers were already complete with no recorded counts; "
            "skipping exact total check.",
            len(completed),
        )


if __name__ == "__main__":
    main()
