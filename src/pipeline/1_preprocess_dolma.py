"""Preprocess sampled Dolma JSONL files: PII replacement, normalization, filtering.

Usage:
    python src/pipeline/1_preprocess_dolma.py --corpus a|b
"""

import argparse
import gzip
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm.auto import tqdm

from pipeline.dolma_config import PII_REPLACEMENTS, specs_for_corpus
from util.paths import DOLMA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimum post-cleaning token counts
MIN_TOKENS = {"a": 100, "b": 50}

# Precompiled regex for horizontal whitespace collapse (preserve newlines)
_H_WHITESPACE = re.compile(r"[^\S\n]+")


def clean_text(text: str) -> str:
    """Apply PII replacement, NFC normalization, and whitespace collapse."""
    # 1. PII placeholder replacement
    for placeholder, substitute in PII_REPLACEMENTS.items():
        text = text.replace(placeholder, substitute)

    # 2. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 3. Horizontal whitespace collapse (preserve newlines as paragraph boundaries)
    text = _H_WHITESPACE.sub(" ", text)

    # Trim leading/trailing whitespace per line, then overall
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()

    return text


def preprocess_corpus(corpus: str) -> None:
    specs = specs_for_corpus(corpus)
    corpus_dir = DOLMA_DIR / f"corpus_{corpus}"
    out_dir = DOLMA_DIR / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"corpus_{corpus}.jsonl.gz"

    min_tok = MIN_TOKENS[corpus]
    total_docs = 0
    total_tokens = 0
    dropped = 0
    source_stats: dict[str, dict] = {}

    logger.info("Preprocessing corpus %s → %s (min tokens: %d)", corpus, out_path, min_tok)

    with gzip.open(out_path, "wt", encoding="utf-8") as gz_out:
        for spec in specs:
            src_path = corpus_dir / f"{spec.name}.jsonl.gz"
            if not src_path.exists():
                logger.warning("Missing source file: %s — skipping", src_path)
                continue

            src_docs = 0
            src_tokens = 0
            src_dropped = 0

            logger.info("  Processing %s …", spec.name)
            with gzip.open(src_path, "rt", encoding="utf-8") as gz_in:
                for line in tqdm(gz_in, desc=spec.name, unit=" docs"):
                    doc = json.loads(line)
                    text = clean_text(doc.get("text", ""))
                    tok_count = len(text.split())

                    # Post-cleaning minimum length filter
                    if tok_count < min_tok:
                        src_dropped += 1
                        continue

                    doc["text"] = text
                    doc["token_count"] = tok_count
                    gz_out.write(json.dumps(doc) + "\n")

                    src_docs += 1
                    src_tokens += tok_count

            source_stats[spec.name] = {
                "docs": src_docs,
                "tokens": src_tokens,
                "dropped": src_dropped,
            }
            total_docs += src_docs
            total_tokens += src_tokens
            dropped += src_dropped

            logger.info("    %s: %s docs, %s tokens (%d dropped)",
                        spec.name, f"{src_docs:,}", f"{src_tokens:,}", src_dropped)

    logger.info("Corpus %s total: %s docs, %s tokens (%d dropped by length filter)",
                corpus, f"{total_docs:,}", f"{total_tokens:,}", dropped)
    logger.info("Output: %s", out_path)

    # Write preprocessing summary alongside the output
    summary_path = out_dir / f"corpus_{corpus}_preprocess_summary.json"
    summary_path.write_text(json.dumps({
        "corpus": corpus,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "dropped_by_length": dropped,
        "min_tokens": min_tok,
        "per_source": source_stats,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Preprocess sampled Dolma JSONL")
    parser.add_argument("--corpus", required=True, choices=["a", "b"])
    args = parser.parse_args()
    preprocess_corpus(args.corpus)


if __name__ == "__main__":
    main()
