"""Verify sampled & preprocessed Dolma corpora: token counts, PII, duplicates.

Usage:
    python src/pipeline/2_verify_dolma.py --corpus a|b
"""

import argparse
import gzip
import json
import logging
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm.auto import tqdm

from pipeline.dolma_config import DEFAULT_SEED, specs_for_corpus
from util.paths import DOLMA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PII_MARKER = "|||"


def verify_corpus(corpus: str, seed: int = DEFAULT_SEED) -> dict:
    specs = specs_for_corpus(corpus)
    preprocessed_path = DOLMA_DIR / "preprocessed" / f"corpus_{corpus}.jsonl.gz"

    if not preprocessed_path.exists():
        logger.error("Preprocessed file not found: %s", preprocessed_path)
        sys.exit(1)

    # -- Load all docs, grouped by source -------------------------------------
    by_source: dict[str, list[dict]] = {s.name: [] for s in specs}
    all_ids: list[str] = []
    pii_count = 0
    empty_count = 0

    logger.info("Loading %s …", preprocessed_path)
    with gzip.open(preprocessed_path, "rt", encoding="utf-8") as gz:
        for line in tqdm(gz, desc="Reading", unit=" docs"):
            doc = json.loads(line)
            src = doc.get("source", "unknown")
            if src in by_source:
                by_source[src].append(doc)
            else:
                by_source.setdefault(src, []).append(doc)

            all_ids.append(doc.get("dolma_id", ""))
            text = doc.get("text", "")
            if PII_MARKER in text:
                pii_count += 1
            if not text.strip():
                empty_count += 1

    # -- Check 1: Token counts per source within ±5% of target ----------------
    issues: list[str] = []
    source_reports: dict[str, dict] = {}

    for spec in specs:
        docs = by_source.get(spec.name, [])
        tok_counts = [d.get("token_count", len(d.get("text", "").split())) for d in docs]
        total_tokens = sum(tok_counts)
        n_docs = len(docs)

        pct_of_target = total_tokens / spec.target_tokens * 100 if spec.target_tokens else 0
        deviation = abs(total_tokens - spec.target_tokens) / spec.target_tokens * 100

        if not tok_counts:
            p5 = p95 = med = mean = 0
        else:
            sorted_tc = sorted(tok_counts)
            p5 = sorted_tc[max(0, int(len(sorted_tc) * 0.05))]
            p95 = sorted_tc[min(len(sorted_tc) - 1, int(len(sorted_tc) * 0.95))]
            med = statistics.median(tok_counts)
            mean = statistics.mean(tok_counts)

        report = {
            "docs": n_docs,
            "total_tokens": total_tokens,
            "target_tokens": spec.target_tokens,
            "pct_of_target": round(pct_of_target, 2),
            "deviation_pct": round(deviation, 2),
            "within_5pct": deviation <= 5,
            "mean_doc_tokens": round(mean, 1),
            "median_doc_tokens": round(med, 1),
            "p5_doc_tokens": p5,
            "p95_doc_tokens": p95,
        }
        source_reports[spec.name] = report

        status = "OK" if deviation <= 5 else "WARNING"
        logger.info("  %s  %s: %s tokens (%.1f%% of %s target, deviation %.1f%%)",
                     status, spec.name, f"{total_tokens:,}", pct_of_target,
                     f"{spec.target_tokens:,}", deviation)

        if deviation > 5:
            issues.append(f"{spec.name}: {deviation:.1f}% deviation from target")

        # Min doc length check
        min_spec = spec.min_doc_tokens
        short_docs = sum(1 for tc in tok_counts if tc < min_spec)
        if short_docs:
            issues.append(f"{spec.name}: {short_docs} docs below min length {min_spec}")
            logger.warning("    %d docs below min length %d", short_docs, min_spec)

    # -- Check 2: No remaining PII markers ------------------------------------
    if pii_count:
        issues.append(f"{pii_count} documents still contain PII markers (|||)")
    logger.info("  PII markers remaining: %d %s", pii_count,
                "OK" if pii_count == 0 else "FAIL")

    # -- Check 3: No empty documents ------------------------------------------
    if empty_count:
        issues.append(f"{empty_count} empty documents")
    logger.info("  Empty documents: %d %s", empty_count,
                "OK" if empty_count == 0 else "FAIL")

    # -- Check 4: No duplicate dolma_id values --------------------------------
    unique_ids = set(all_ids)
    dup_count = len(all_ids) - len(unique_ids)
    if dup_count:
        issues.append(f"{dup_count} duplicate dolma_id values")
    logger.info("  Duplicate IDs: %d %s", dup_count,
                "OK" if dup_count == 0 else "FAIL")

    # -- Random sample for manual inspection ----------------------------------
    logger.info("\n=== Random sample (10 docs per source) ===")
    rng = random.Random(seed)
    sample_records: dict[str, list[dict]] = {}
    for spec in specs:
        docs = by_source.get(spec.name, [])
        k = min(10, len(docs))
        sampled = rng.sample(docs, k) if docs else []
        sample_records[spec.name] = []
        logger.info("\n--- %s (%d sampled) ---", spec.name, k)
        for i, doc in enumerate(sampled):
            preview = doc.get("text", "")[:200].replace("\n", " ")
            logger.info("  [%d] id=%s  tokens=%d\n       %s …",
                        i, doc.get("dolma_id", "?"), doc.get("token_count", 0), preview)
            sample_records[spec.name].append({
                "dolma_id": doc.get("dolma_id"),
                "token_count": doc.get("token_count"),
                "text_preview": doc.get("text", "")[:300],
            })

    # -- Write verification report --------------------------------------------
    report = {
        "corpus": corpus,
        "total_docs": sum(len(v) for v in by_source.values()),
        "total_tokens": sum(r["total_tokens"] for r in source_reports.values()),
        "pii_remaining": pii_count,
        "empty_docs": empty_count,
        "duplicate_ids": dup_count,
        "issues": issues,
        "passed": len(issues) == 0,
        "per_source": source_reports,
        "sample_docs": sample_records,
    }

    log_dir = DOLMA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("\nVerification report written to %s", report_path)

    if issues:
        logger.warning("\nISSUES FOUND:")
        for issue in issues:
            logger.warning("  - %s", issue)
    else:
        logger.info("\nAll checks passed.")

    return report


def main():
    parser = argparse.ArgumentParser(description="Verify Dolma corpus quality")
    parser.add_argument("--corpus", required=True, choices=["a", "b"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    verify_corpus(args.corpus, seed=args.seed)


if __name__ == "__main__":
    main()
