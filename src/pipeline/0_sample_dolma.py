"""Sample register-contrastive corpora from Dolma v1.7.

Usage:
    python src/pipeline/0_sample_dolma.py --corpus a|b [--source NAME] [--seed N] [--resume]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.dolma_config import DEFAULT_SEED, specs_for_corpus
from pipeline.dolma_stream import sample_source
from util.paths import DOLMA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sample from Dolma v1.7")
    parser.add_argument("--corpus", required=True, choices=["a", "b"],
                        help="Which corpus to sample (a=edited, b=unedited)")
    parser.add_argument("--source", default=None,
                        help="Sample only this source (e.g. 'wiki')")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    specs = specs_for_corpus(args.corpus)
    if args.source:
        specs = [s for s in specs if s.name == args.source]
        if not specs:
            parser.error(f"Unknown source {args.source!r} for corpus {args.corpus}")

    log_dir = DOLMA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in specs:
        logger.info("=== Sampling %s (target %s tokens) ===",
                     spec.name, f"{spec.target_tokens:,}")
        state = sample_source(spec, seed=args.seed, resume=args.resume)
        results.append({
            "source": state.source_name,
            "corpus": spec.corpus,
            "target_tokens": spec.target_tokens,
            "sampled_tokens": state.accumulated_tokens,
            "sampled_docs": state.accumulated_docs,
            "files_processed": len(state.completed_files),
            "finished": state.finished,
        })

    # Write reproducibility log
    sampling_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus,
        "seed": args.seed,
        "resumed": args.resume,
        "sources": results,
    }
    log_path = log_dir / "sampling_log.json"
    # Append to existing log if present
    if log_path.exists():
        existing = json.loads(log_path.read_text())
        if isinstance(existing, list):
            existing.append(sampling_log)
        else:
            existing = [existing, sampling_log]
        log_path.write_text(json.dumps(existing, indent=2))
    else:
        log_path.write_text(json.dumps([sampling_log], indent=2))

    logger.info("Sampling log written to %s", log_path)

    # Summary
    for r in results:
        pct = r["sampled_tokens"] / r["target_tokens"] * 100 if r["target_tokens"] else 0
        logger.info("  %s: %s tokens (%.1f%% of target), %s docs",
                     r["source"], f"{r['sampled_tokens']:,}", pct,
                     f"{r['sampled_docs']:,}")


if __name__ == "__main__":
    main()
