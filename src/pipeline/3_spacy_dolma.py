"""Process preprocessed Dolma JSONL through spaCy and save DocBins.

Streams gzipped JSONL → nlp.pipe() → DocBins split by token count.
Supports checkpoint/resume to survive interruptions.

Usage:
    python src/pipeline/3_spacy_dolma.py --corpus a|b [--resume] \
        [--n-process 32] [--batch-size 512] [--tokens-per-file 10000000]
"""

import argparse
import gzip
import json
import logging
import sys
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
    """Load the preprocessing summary for a corpus."""
    path = DOLMA_DIR / "preprocessed" / f"corpus_{corpus}_preprocess_summary.json"
    with open(path) as f:
        return json.load(f)


def read_checkpoint(checkpoint_path: Path) -> dict:
    """Read checkpoint state, or return defaults."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"lines_consumed": 0, "docbins_saved": 0}


def write_checkpoint(checkpoint_path: Path, lines_consumed: int, docbins_saved: int):
    """Atomically write checkpoint."""
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "lines_consumed": lines_consumed,
        "docbins_saved": docbins_saved,
    }, indent=2))
    tmp.rename(checkpoint_path)


def stream_texts(corpus: str, skip_lines: int = 0):
    """Yield (text, metadata_dict) from preprocessed JSONL, optionally skipping lines."""
    input_path = DOLMA_DIR / "preprocessed" / f"corpus_{corpus}.jsonl.gz"
    with gzip.open(input_path, "rt", encoding="utf-8") as gz:
        for i, line in enumerate(gz):
            if i < skip_lines:
                continue
            doc = json.loads(line)
            meta = {"dolma_id": doc["dolma_id"], "source": doc["source"]}
            yield (doc["text"], meta)


def process_corpus(
    corpus: str,
    resume: bool = False,
    n_process: int = 32,
    batch_size: int = 512,
    tokens_per_file: int = 10_000_000,
):
    summary = load_summary(corpus)
    expected_docs = summary["total_docs"]
    expected_tokens = summary["total_tokens"]
    logger.info(
        "Corpus %s: expecting %s docs, ~%s tokens",
        corpus, f"{expected_docs:,}", f"{expected_tokens:,}",
    )

    output_dir = DOLMA_DOCBINS_DIR / f"corpus_{corpus}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = DOLMA_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"spacy_{corpus}.json"

    # Resume handling
    skip_lines = 0
    docbin_idx = 0
    if resume:
        ckpt = read_checkpoint(checkpoint_path)
        skip_lines = ckpt["lines_consumed"]
        docbin_idx = ckpt["docbins_saved"]
        if skip_lines > 0:
            logger.info(
                "Resuming: skipping %s lines, starting at DocBin %04d",
                f"{skip_lines:,}", docbin_idx,
            )

    # Load spaCy model
    logger.info("Loading spaCy model (NER disabled)...")
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.max_length = 5_000_000

    # Stream and process
    remaining = expected_docs - skip_lines
    text_stream = stream_texts(corpus, skip_lines=skip_lines)

    current_docs = []
    current_tokens = 0
    lines_consumed = skip_lines
    total_docs_saved = 0

    logger.info(
        "Starting spaCy processing: n_process=%d, batch_size=%d, tokens_per_file=%s",
        n_process, batch_size, f"{tokens_per_file:,}",
    )

    with tqdm(total=remaining, desc=f"spaCy corpus_{corpus}", unit=" docs") as pbar:
        for doc, meta in nlp.pipe(
            text_stream,
            as_tuples=True,
            n_process=n_process,
            batch_size=batch_size,
        ):
            doc.user_data["meta"] = meta
            current_docs.append(doc)
            current_tokens += len(doc)
            lines_consumed += 1

            # Flush DocBin when token threshold is exceeded
            if current_tokens >= tokens_per_file:
                output_path = output_dir / f"processed_docs_{docbin_idx:04d}.spacy"
                docbin = DocBin(docs=current_docs, store_user_data=True)
                docbin.to_disk(output_path)
                total_docs_saved += len(current_docs)

                logger.info(
                    "Saved DocBin %04d: %s docs, %s tokens → %s",
                    docbin_idx,
                    f"{len(current_docs):,}",
                    f"{current_tokens:,}",
                    output_path.name,
                )

                docbin_idx += 1
                current_docs = []
                current_tokens = 0

                # Checkpoint after each flush
                write_checkpoint(checkpoint_path, lines_consumed, docbin_idx)

            pbar.update(1)

    # Save final DocBin
    if current_docs:
        output_path = output_dir / f"processed_docs_{docbin_idx:04d}.spacy"
        docbin = DocBin(docs=current_docs, store_user_data=True)
        docbin.to_disk(output_path)
        total_docs_saved += len(current_docs)

        logger.info(
            "Saved final DocBin %04d: %s docs, %s tokens → %s",
            docbin_idx,
            f"{len(current_docs):,}",
            f"{current_tokens:,}",
            output_path.name,
        )
        docbin_idx += 1
        write_checkpoint(checkpoint_path, lines_consumed, docbin_idx)

    # Verification
    logger.info("--- Verification ---")
    total_docbin_files = len(list(output_dir.glob("*.spacy")))
    logger.info("Total DocBin files on disk: %d", total_docbin_files)
    logger.info("Total docs saved this run: %s", f"{total_docs_saved:,}")

    # Count docs across all DocBin files
    actual_total = 0
    for spacy_file in sorted(output_dir.glob("*.spacy")):
        db = DocBin().from_disk(spacy_file)
        actual_total += len(db)

    logger.info(
        "Verification: %s docs across all DocBins (expected %s)",
        f"{actual_total:,}", f"{expected_docs:,}",
    )
    if actual_total == expected_docs:
        logger.info("PASS: doc count matches preprocess summary")
    else:
        logger.warning(
            "MISMATCH: DocBin total (%d) != expected (%d) — delta %d",
            actual_total, expected_docs, actual_total - expected_docs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Process Dolma corpus through spaCy and save DocBins"
    )
    parser.add_argument("--corpus", required=True, choices=["a", "b"])
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--n-process", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--tokens-per-file", type=int, default=10_000_000,
        help="Approximate spaCy token count per DocBin file",
    )
    args = parser.parse_args()

    process_corpus(
        corpus=args.corpus,
        resume=args.resume,
        n_process=args.n_process,
        batch_size=args.batch_size,
        tokens_per_file=args.tokens_per_file,
    )


if __name__ == "__main__":
    main()
