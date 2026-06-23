"""Configuration for Dolma v1.7 sampling pipeline."""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dolma v1.7 remote layout
# ---------------------------------------------------------------------------
DOLMA_BASE_URL = "https://olmo-data.org/dolma-v1_7"

# Maps our short source name -> list of directory prefixes in the Dolma URL
# index.  Each Dolma file path looks like  <dir>/xxxx.json.gz
SOURCE_DIRS: dict[str, list[str]] = {
    "wiki": ["wiki"],
    "pes2o": ["pes2o"],
    "cc_news": ["cc_news_head", "cc_news_middle", "cc_news_tail"],
    "reddit": ["reddit"],
    "stackexchange": ["redpajama-stackexchange"],
    "cc_blogs": ["cc_en_head"],      # blog subset identified by domain filter
}

# ---------------------------------------------------------------------------
# SourceSpec — one per sampling target
# ---------------------------------------------------------------------------
@dataclass
class SourceSpec:
    """Describes a single source to sample from Dolma."""
    name: str
    corpus: str                       # "a" or "b"
    target_tokens: int                # whitespace-token target
    min_doc_tokens: int = 50          # minimum whitespace tokens per doc
    domain_filter: list[str] = field(default_factory=list)
    max_code_fraction: float | None = None  # e.g. 0.5 for StackExchange

# ---------------------------------------------------------------------------
# Corpus A (Edited)
# ---------------------------------------------------------------------------
CORPUS_A_SPECS: list[SourceSpec] = [
    SourceSpec(
        name="wiki",
        corpus="a",
        target_tokens=450_000_000,
        min_doc_tokens=200,
    ),
    SourceSpec(
        name="pes2o",
        corpus="a",
        target_tokens=350_000_000,
    ),
    SourceSpec(
        name="cc_news",
        corpus="a",
        target_tokens=200_000_000,
        # No domain filter — cc_news_head is already curated news content
    ),
]

# ---------------------------------------------------------------------------
# Corpus B (Unedited)
# ---------------------------------------------------------------------------
BLOG_DOMAINS: list[str] = [
    "wordpress.com",
    "blogspot.com",
    "medium.com",
    "substack.com",
    "tumblr.com",
    "livejournal.com",
    "ghost.io",
    "hashnode.dev",
    "dev.to",
]

CORPUS_B_SPECS: list[SourceSpec] = [
    SourceSpec(
        name="reddit",
        corpus="b",
        target_tokens=600_000_000,
        min_doc_tokens=50,
    ),
    SourceSpec(
        name="stackexchange",
        corpus="b",
        target_tokens=200_000_000,
        min_doc_tokens=50,
        max_code_fraction=0.5,
    ),
    SourceSpec(
        name="cc_blogs",
        corpus="b",
        target_tokens=200_000_000,
        min_doc_tokens=50,
        domain_filter=BLOG_DOMAINS,
    ),
]

# ---------------------------------------------------------------------------
# PII placeholder replacement
# ---------------------------------------------------------------------------
# Dolma uses  |||EMAIL_ADDRESS|||  |||PHONE_NUMBER|||  |||IP_ADDRESS|||
# Replace with linguistically parseable substitutes so spaCy/dep parsers
# produce valid trees.
PII_REPLACEMENTS: dict[str, str] = {
    "|||EMAIL_ADDRESS|||": "email@example.com",
    "|||PHONE_NUMBER|||": "555-0100",
    "|||IP_ADDRESS|||": "0.0.0.0",
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
OVERSHOOT_FACTOR = 1.05  # sample 5% extra to absorb preprocessing losses


def specs_for_corpus(corpus: str) -> list[SourceSpec]:
    """Return the list of SourceSpecs for corpus 'a' or 'b'."""
    if corpus == "a":
        return CORPUS_A_SPECS
    elif corpus == "b":
        return CORPUS_B_SPECS
    else:
        raise ValueError(f"Unknown corpus: {corpus!r} (expected 'a' or 'b')")
