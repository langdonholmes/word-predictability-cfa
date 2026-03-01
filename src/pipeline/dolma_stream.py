"""Streaming engine for sampling from Dolma v1.7 without local storage."""

import gzip
import hashlib
import json
import logging
import random
import shutil
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from tqdm.auto import tqdm

from pipeline.dolma_config import (
    DOLMA_BASE_URL,
    OVERSHOOT_FACTOR,
    SOURCE_DIRS,
    SourceSpec,
)
from util.paths import DOLMA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

class DolmaStreamer:
    """Stream a remote .json.gz file: HTTP chunked -> gzip decompress -> JSONL parse.

    Yields ``(doc_dict, whitespace_token_count)`` tuples.
    """

    CHUNK_SIZE = 1 << 18  # 256 KiB

    def __init__(self, url: str, retries: int = 3):
        self.url = url
        self.retries = retries

    def __iter__(self):
        for attempt in range(1, self.retries + 1):
            try:
                yield from self._stream()
                return
            except (requests.RequestException, zlib.error, json.JSONDecodeError) as exc:
                if attempt == self.retries:
                    raise
                wait = 2 ** attempt
                logger.warning("Retry %d/%d for %s (%s), sleeping %ds",
                               attempt, self.retries, self.url, exc, wait)
                time.sleep(wait)

    def _stream(self):
        resp = requests.get(self.url, stream=True, timeout=120)
        resp.raise_for_status()
        decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)  # gzip mode
        buf = b""
        for chunk in resp.iter_content(chunk_size=self.CHUNK_SIZE):
            buf += decompressor.decompress(chunk)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("text", "")
                tok_count = len(text.split())
                yield doc, tok_count
        # flush
        buf += decompressor.flush()
        for line in buf.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text = doc.get("text", "")
            tok_count = len(text.split())
            yield doc, tok_count

# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------

@dataclass
class SamplingState:
    """Tracks progress for a single source across interruptions."""
    source_name: str
    accumulated_tokens: int = 0
    accumulated_docs: int = 0
    completed_files: list[str] = field(default_factory=list)
    finished: bool = False

    # -- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> "SamplingState":
        return cls(**json.loads(path.read_text()))

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _passes_length_filter(tok_count: int, spec: SourceSpec) -> bool:
    return tok_count >= spec.min_doc_tokens


def _passes_domain_filter(doc: dict, spec: SourceSpec) -> bool:
    if not spec.domain_filter:
        return True
    # Dolma CC docs have a "metadata" dict that may include source_domain or url
    meta = doc.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    domain = meta.get("source_domain", "")
    url = meta.get("url", doc.get("url", ""))
    for d in spec.domain_filter:
        if d in domain or d in url:
            return True
    return False


def _passes_code_filter(doc: dict, spec: SourceSpec) -> bool:
    if spec.max_code_fraction is None:
        return True
    text = doc.get("text", "")
    lines = text.splitlines()
    if not lines:
        return False
    code_chars = 0
    total_chars = 0
    for line in lines:
        stripped = line.strip()
        n = len(stripped)
        total_chars += n
        # Heuristic: line starts with common code indicators or is indented 4+
        if (stripped.startswith(("{", "}", "<", "//", "/*", "#include", "import ",
                                "def ", "class ", "function ", "var ", "let ",
                                "const ", "return ", "if (", "for (", "while ("))
                or line.startswith("    ")
                or line.startswith("\t")):
            code_chars += n
    if total_chars == 0:
        return False
    return (code_chars / total_chars) <= spec.max_code_fraction


def _accept_by_hash(doc_id: str, seed: int, threshold: int) -> bool:
    """Deterministic acceptance: md5(doc_id + seed) mod 10000 < threshold."""
    h = hashlib.md5(f"{doc_id}{seed}".encode()).hexdigest()
    return int(h, 16) % 10000 < threshold

# ---------------------------------------------------------------------------
# URL listing
# ---------------------------------------------------------------------------

def _get_file_urls(source_name: str) -> list[str]:
    """Get URLs for all .json.gz files belonging to *source_name*.

    Downloads the Dolma URL manifest via ``huggingface_hub``, then filters
    to lines whose path starts with one of the configured directory prefixes.
    """
    from huggingface_hub import hf_hub_download

    manifest_path = hf_hub_download(
        repo_id="allenai/dolma",
        filename="urls/v1_7.txt",
        repo_type="dataset",
    )
    dirs = SOURCE_DIRS[source_name]
    urls: list[str] = []
    with open(manifest_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            # Each line is a full URL like https://olmo-data.org/dolma-v1_7/wiki/...
            # Extract the relative path after the base.
            for d in dirs:
                if f"/{d}/" in line:
                    urls.append(line)
                    break
    return urls

# ---------------------------------------------------------------------------
# Main sampling function
# ---------------------------------------------------------------------------

def sample_source(
    spec: SourceSpec,
    seed: int,
    resume: bool = False,
) -> SamplingState:
    """Sample documents from a single Dolma source.

    Streams remote gzipped JSONL files, filters, and writes accepted docs
    to ``data/dolma/corpus_{a|b}/{source}.jsonl.gz``.
    """
    corpus_dir = DOLMA_DIR / f"corpus_{spec.corpus}"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = DOLMA_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{spec.name}.checkpoint.json"

    # -- resume or fresh start -------------------------------------------------
    if resume and checkpoint_path.exists():
        state = SamplingState.load(checkpoint_path)
        if state.finished:
            logger.info("Source %s already finished (%s tokens, %s docs)",
                        spec.name, f"{state.accumulated_tokens:,}",
                        f"{state.accumulated_docs:,}")
            return state
        logger.info("Resuming %s from checkpoint (%s tokens, %s docs, %d files done)",
                     spec.name, f"{state.accumulated_tokens:,}",
                     f"{state.accumulated_docs:,}", len(state.completed_files))
    else:
        state = SamplingState(source_name=spec.name)

    # -- get & shuffle file URLs ----------------------------------------------
    logger.info("Fetching URL list for %s …", spec.name)
    all_urls = _get_file_urls(spec.name)
    rng = random.Random(seed)
    rng.shuffle(all_urls)
    logger.info("Found %d files for %s", len(all_urls), spec.name)

    # -- determine hash-acceptance threshold -----------------------------------
    # Start with a generous threshold; exact ratio doesn't matter much because
    # we stop on the token budget anyway.  A threshold of 10000 means "accept all".
    threshold = 10000  # accept everything by default

    target_with_buffer = int(spec.target_tokens * OVERSHOOT_FACTOR)
    output_path = corpus_dir / f"{spec.name}.jsonl.gz"

    # Temporary directory for per-file JSONLs (concatenated at end)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"dolma_{spec.name}_"))
    try:
        pending_urls = [u for u in all_urls if u not in set(state.completed_files)]
        pbar = tqdm(total=target_with_buffer, initial=state.accumulated_tokens,
                    unit="tok", desc=spec.name, unit_scale=True)

        for file_url in pending_urls:
            if state.accumulated_tokens >= target_with_buffer:
                break

            tmp_file = tmp_dir / f"{len(state.completed_files):06d}.jsonl"
            docs_in_file = 0
            toks_in_file = 0

            with open(tmp_file, "w") as fout:
                for doc, tok_count in DolmaStreamer(file_url):
                    if not _passes_length_filter(tok_count, spec):
                        continue
                    if not _passes_domain_filter(doc, spec):
                        continue
                    if not _passes_code_filter(doc, spec):
                        continue

                    doc_id = doc.get("id", doc.get("dolma_id", ""))
                    if not _accept_by_hash(doc_id, seed, threshold):
                        continue

                    # Write accepted document
                    out_record = {
                        "dolma_id": doc_id,
                        "source": spec.name,
                        "text": doc.get("text", ""),
                        "token_count": tok_count,
                        "metadata": doc.get("metadata", {}),
                    }
                    fout.write(json.dumps(out_record) + "\n")
                    docs_in_file += 1
                    toks_in_file += tok_count

                    if state.accumulated_tokens + toks_in_file >= target_with_buffer:
                        break

            state.accumulated_tokens += toks_in_file
            state.accumulated_docs += docs_in_file
            state.completed_files.append(file_url)
            state.save(checkpoint_path)
            pbar.update(toks_in_file)

            logger.debug("Finished file %s: +%d docs, +%s tokens",
                         file_url.rsplit("/", 1)[-1], docs_in_file,
                         f"{toks_in_file:,}")

        pbar.close()

        # -- concatenate temp files into final gzipped JSONL -------------------
        logger.info("Concatenating %d temp files → %s",
                     len(state.completed_files), output_path)
        with gzip.open(output_path, "wt", encoding="utf-8") as gz_out:
            for tmp_file in sorted(tmp_dir.iterdir()):
                if not tmp_file.name.endswith(".jsonl"):
                    continue
                with open(tmp_file) as fin:
                    for line in fin:
                        gz_out.write(line)

        state.finished = True
        state.save(checkpoint_path)
        logger.info("Done: %s — %s tokens, %s docs",
                     spec.name, f"{state.accumulated_tokens:,}",
                     f"{state.accumulated_docs:,}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return state
