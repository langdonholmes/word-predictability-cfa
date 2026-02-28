# Dolma Sampling Plan: Two Register-Based Reference Corpora for Collocational Analysis

## Overview

This plan describes the construction of two 1-billion-token reference corpora from Dolma v1.7 (Allen AI), designed to support dependency-based collocational analysis for L2 writing proficiency research. The corpora are split along a multi-author/edited vs. single-author/unedited dimension, providing register-contrastive collocational norms.

**Source dataset:** [allenai/dolma](https://huggingface.co/datasets/allenai/dolma) (v1_7, default version)
**License:** ODC-By 1.0
**Format:** Gzipped JSONL, one document per line, with `text` and `metadata` fields

---

## Corpus Design

### Corpus A — Multi-Author / Edited (1B tokens)

Texts that have undergone editorial review, institutional oversight, or collaborative revision. Collocational patterns here reflect conventionalized, normative English usage.

| Source | Sample Size | Available | Sampling Rate | Register Character |
|---|---|---|---|---|
| Wikipedia + Wikibooks | 400M tokens | ~3.7B tokens | ~10.8% | Encyclopedic expository; collaboratively edited; neutral tone; broad topical coverage |
| peS2o | 350M tokens | ~57B tokens | ~0.6% | Peer-reviewed academic prose; formal register; STEM-heavy but includes social sciences and humanities |
| CC News | 150M tokens | ~14.3B tokens | ~1.0% | Journalistic prose; edited; semi-formal; narrative and informational |
| Project Gutenberg | 100M tokens | ~5.3B tokens | ~1.9% | Literary prose; formal; pre-1928 public domain; archaic collocations possible |
| **Total** | **1B tokens** | | | |

### Corpus B — Single-Author / Unedited (1B tokens)

Texts produced by individual writers without formal editorial processes. Collocational patterns here reflect naturalistic, spontaneous English usage.

| Source | Sample Size | Available | Sampling Rate | Register Character |
|---|---|---|---|---|
| Reddit | 600M tokens | ~79.9B tokens | ~0.75% | Informal conversational; single-author posts; wide topical range; unedited |
| StackExchange | 200M tokens | ~19.6B tokens | ~1.0% | Technical Q&A; single-author posts within multi-voice threads; semi-formal |
| Blog-platform web text | 200M tokens | TBD (URL-filtered from CC) | TBD | Personal/opinion writing; single-author; variable formality; narrative and expository |
| **Total** | **1B tokens** | | | |

---

## Data Access

Dolma v1.7 is distributed as gzipped JSONL files organized by source. To download:

```bash
# Clone the repository to get URL lists
git clone https://huggingface.co/datasets/allenai/dolma

# Download specific source subsets using the URL lists
# Each source has its own set of file URLs in dolma/urls/v1_7.txt
# Filter the URL list by source prefix before downloading

# Example: download only Wikipedia files
grep "wiki" dolma/urls/v1_7.txt | xargs -n 1 -P 8 wget -q -P ./dolma_data/wiki/
```

Each JSONL line has the structure:

```json
{
  "text": "The full document text...",
  "metadata": {
    "source": "wiki",
    "url": "https://en.wikipedia.org/wiki/...",
    ...
  }
}
```

Source identifiers in v1.7 file paths include: `wiki`, `cc` (Common Crawl), `refined-web`, `c4`, `reddit`, `starcoder`, `pes2o`, `arxiv`, `stackexchange`, `flan`, `cc-news`, `open-web-math`, `algebraic-stack`, `gutenberg`, `megawika`.

---

## Sampling Procedures

### General Approach

For each source, the sampling procedure is:

1. Stream through the gzipped JSONL files for that source.
2. Apply any source-specific filters (see below).
3. Count tokens using a consistent tokenizer (recommendation: the GPT-2/LLaMA tokenizer, which Dolma uses for its reported token counts, or spaCy's English tokenizer for consistency with downstream parsing).
4. Randomly sample documents until the target token count is reached.
5. Use reservoir sampling or a two-pass approach (first pass: count total documents and tokens; second pass: sample with known probability).

**Important:** Token counts in Dolma's documentation use the LLaMA tokenizer. If you use a different tokenizer (e.g., spaCy whitespace-based), your actual word counts will differ. Calibrate your target accordingly. As a rough heuristic, 1B LLaMA tokens ≈ 750M whitespace-delimited words.

### Source-Specific Notes

**Wikipedia + Wikibooks**
- Files are in the `wiki` subset of Dolma v1.7.
- Wikipedia dominates; Wikibooks is a small fraction.
- No special filtering needed beyond random sampling.
- Consider stratifying by article length to avoid oversampling stubs (exclude documents under 200 tokens).

**peS2o (Semantic Scholar)**
- Files are in the `pes2o` subset.
- Contains 38M permissively licensed academic manuscripts.
- Skews toward STEM disciplines. If disciplinary balance matters, metadata may include venue/field information that could support stratified sampling (verify in the actual files).
- Documents may contain artifacts from PDF extraction (garbled tables, inline citations, reference sections). Consider truncating documents at the reference section if detectable.

**CC News**
- Files are in the `cc-news` subset (new in v1.7).
- Processed through Dolma's quality filtering pipeline.
- Random sampling should be sufficient; news articles are relatively uniform in length.

**Project Gutenberg**
- Files are in the `gutenberg` subset.
- Only ~56,000 English-language books are available (~5.3B tokens total).
- Sample 100M tokens ≈ roughly 1,000 books at ~100K tokens each.
- Consider excluding very short documents (poems, pamphlets) and very long documents (multi-volume works) to get a representative sample of prose fiction and nonfiction.
- **Caveat:** Pre-1928 literary English. Archaic collocations (e.g., "make haste," "take leave") will appear. At 10% of Corpus A, these will have limited impact on aggregate collocational norms, but document this as a known characteristic.

**Reddit**
- Files are in the `reddit` subset (sourced from PushShift).
- Individual posts/comments are documents.
- Very short documents are common (one-line replies). Apply a minimum length threshold (e.g., ≥50 tokens) to ensure enough syntactic context for dependency parsing.
- Subreddit metadata may or may not be preserved in Dolma's version. If available, this enables stratified sampling across topical communities.

**StackExchange**
- Files are in the `stackexchange` subset (via RedPajama v1).
- Contains both questions and answers as separate documents.
- Apply a minimum length threshold (≥50 tokens) as with Reddit.
- Includes many code-heavy posts (especially from Stack Overflow). Consider filtering posts that are majority code (e.g., if >50% of lines contain common code indicators like braces, semicolons, indentation patterns). Alternatively, sample more heavily from non-SO StackExchange sites (e.g., English Language & Usage, Academia, Writing) for more prose-heavy content.

**Blog-Platform Web Text**
- Requires filtering the Common Crawl portion of Dolma (`cc` and/or `refined-web` subsets) by URL domain.
- Target domains (case-insensitive substring match on URL):
  - `wordpress.com`
  - `blogspot.com`
  - `medium.com`
  - `substack.com`
  - `tumblr.com`
  - `livejournal.com`
  - `ghost.io`
  - `hashnode.dev`
  - `dev.to` (technical blogs, semi-formal)
- Implementation: stream through CC files, check URL field in metadata, collect matching documents.
- This is the most engineering-intensive step. If the overhead is not justified, an alternative is to increase Reddit to 700M and StackExchange to 300M and skip blog extraction entirely.
- **Fallback option:** If blog filtering proves difficult, the MegaWika subset (~4.6B tokens of web pages cited from Wikipedia) could substitute. These are editorially curated web pages, but they lean institutional/multi-author, so they would fit better in Corpus A than Corpus B.

---

## Preprocessing Pipeline

After sampling, apply the following preprocessing steps before dependency parsing:

### 1. PII Token Replacement

Dolma replaces detected PII with placeholder tokens:
- `|||EMAIL_ADDRESS|||`
- `|||IP_ADDRESS|||`
- `|||PHONE_NUMBER|||`

These will produce spurious dependency relations. Replace them with generic nouns before parsing:

```python
import re

pii_map = {
    "|||EMAIL_ADDRESS|||": "someone@example.com",  # or simply "email"
    "|||IP_ADDRESS|||": "0.0.0.0",                  # or simply "address"
    "|||PHONE_NUMBER|||": "000-000-0000",            # or simply "number"
}

def clean_pii(text):
    for placeholder, replacement in pii_map.items():
        text = text.replace(placeholder, replacement)
    return text
```

### 2. Document Boundary Preservation

Each JSONL line is one document. Maintain document boundaries throughout the pipeline so that:
- Dependency parsing does not cross document boundaries.
- Collocational statistics are not computed across documents.

Recommendation: Write each sampled document as a separate file, or use a clear delimiter (e.g., blank line + document ID) in a consolidated file.

### 3. Encoding and Whitespace Normalization

- Ensure UTF-8 encoding throughout.
- Normalize Unicode (NFC form).
- Collapse multiple whitespace characters to single spaces.
- Strip leading/trailing whitespace from documents.

### 4. Minimum Document Length Filter

Apply after PII cleaning:
- **Corpus A:** Minimum 100 tokens per document (academic papers and Wikipedia articles are typically long; this filters extraction artifacts).
- **Corpus B:** Minimum 50 tokens per document (Reddit/StackExchange posts are naturally shorter; a lower threshold preserves the register while excluding one-line replies).

---

## Dependency Parsing

### Parser Selection

| Parser | Model | Speed (approx.) | Accuracy (UAS/LAS) | Notes |
|---|---|---|---|---|
| spaCy | `en_core_web_trf` | ~2K sents/sec (GPU) | 95.1/93.4 | Transformer-based; highest accuracy; requires GPU |
| spaCy | `en_core_web_lg` | ~10K sents/sec (CPU) | 91.9/89.8 | Statistical; fast; no GPU needed |
| Stanza | default English | ~1K sents/sec (GPU) | 94.0/91.8 | Neural; good accuracy; slower than spaCy |

At 1B tokens per corpus (~40M sentences), estimated parsing times:
- `en_core_web_trf` on GPU: ~5.5 hours per corpus
- `en_core_web_lg` on CPU (multicore): ~1.1 hours per corpus
- Stanza on GPU: ~11 hours per corpus

These estimates assume no I/O bottlenecks. Actual throughput will depend on batch size, document length distribution, and hardware. For 2B tokens total, even the transformer model is feasible within a day on a single GPU.

### Output Format

Write parsed output in CoNLL-U format for interoperability:

```
# doc_id = wiki_00001
# text = The cat sat on the mat.
1	The	the	DET	DT	_	2	det	_	_
2	cat	cat	NOUN	NN	_	3	nsubj	_	_
3	sat	sit	VERB	VBD	_	0	root	_	_
4	on	on	ADP	IN	_	6	case	_	_
5	the	the	DET	DT	_	6	det	_	_
6	mat	mat	NOUN	NN	_	3	obl	_	_
7	.	.	PUNCT	.	_	3	punct	_	_
```

This format supports downstream extraction of dependency-based collocations (e.g., all nsubj-verb pairs, verb-dobj pairs, amod-noun pairs).

---

## Collocational Statistics

Once parsed, extract dependency-based collocations and compute association measures. Standard measures include:

- **PMI (Pointwise Mutual Information):** log2(P(w1,w2) / P(w1)P(w2)). Sensitive to low-frequency pairs; consider a minimum co-occurrence threshold (e.g., ≥5).
- **Log-Dice:** 14 + log2(2 * f(w1,w2) / (f(w1) + f(w2))). Less sensitive to corpus size than PMI; bounded; recommended for cross-corpus comparison.
- **Delta P (ΔP):** Directional association; P(w2|w1) - P(w2|¬w1). Useful for capturing asymmetric collocational strength.
- **MI² or MI³:** Variants of PMI that upweight higher-frequency pairs.

For cross-corpus comparison (Corpus A vs. Corpus B), Log-Dice is particularly appropriate because its scale is independent of corpus size, making direct comparison between equal-sized corpora straightforward.

---

## Validation

### Internal Checks

- Verify token counts per source match targets (±5%).
- Confirm no document duplication across or within corpora (spot-check with MinHash or exact-match on random samples).
- Check that PII placeholders have been removed (grep for `|||`).
- Inspect a random sample of 100 documents per source for quality (readability, language, completeness).

### External Benchmarking

- Compare high-frequency collocational rankings (top 100 verb-noun, adjective-noun pairs) against COCA web interface as a sanity check.
- Verify that register-distinctive collocations pattern as expected (e.g., "conduct research" should be stronger in Corpus A; "gonna be" should be stronger in Corpus B).

---

## File Organization

```
dolma-corpora/
├── corpus-a-edited/
│   ├── raw/                    # Sampled JSONL files by source
│   │   ├── wikipedia.jsonl.gz
│   │   ├── pes2o.jsonl.gz
│   │   ├── cc-news.jsonl.gz
│   │   └── gutenberg.jsonl.gz
│   ├── cleaned/                # After preprocessing
│   │   └── corpus-a.jsonl.gz
│   ├── parsed/                 # CoNLL-U output
│   │   └── corpus-a.conllu.gz
│   └── collocations/           # Extracted association measures
│       └── corpus-a-collocations.tsv
├── corpus-b-unedited/
│   ├── raw/
│   │   ├── reddit.jsonl.gz
│   │   ├── stackexchange.jsonl.gz
│   │   └── blogs.jsonl.gz      # URL-filtered from CC (optional)
│   ├── cleaned/
│   │   └── corpus-b.jsonl.gz
│   ├── parsed/
│   │   └── corpus-b.conllu.gz
│   └── collocations/
│       └── corpus-b-collocations.tsv
├── scripts/
│   ├── sample_dolma.py
│   ├── preprocess.py
│   ├── parse_corpus.py
│   └── extract_collocations.py
├── metadata/
│   ├── sampling-log.json       # Exact file sources, RNG seeds, document counts
│   └── corpus-statistics.json  # Final token/document/sentence counts
└── README.md
```

---

## Reproducibility

Record and report the following:
- Dolma version (v1_7) and download date.
- Random seed(s) used for sampling.
- Exact document IDs (URLs or Dolma internal IDs) included in each corpus.
- Tokenizer used for token counting and the mapping between token counts and word counts.
- spaCy/Stanza model version and any configuration parameters.
- All preprocessing steps and parameters (minimum document length, PII replacement strings, etc.).

---

## Open Questions

1. **Blog extraction cost-benefit.** Is 200M tokens of blog text worth the engineering effort of URL-filtering the CC portion? The fallback (Reddit 700M + StackExchange 300M) is simpler and still defensible.

2. **peS2o disciplinary balance.** Should academic papers be stratified by field, or is random sampling acceptable? If the collocational norms will be used as a general reference, disciplinary balance may not matter. If they will be compared against L2 writing in specific fields, it could.

3. **Gutenberg inclusion.** At 100M tokens (10% of Corpus A), Gutenberg adds literary register diversity but introduces pre-1928 collocational patterns. An alternative is to replace it with 100M additional tokens from Wikipedia or CC News.

4. **StackExchange code filtering.** How aggressively to filter code-heavy posts? Removing all posts with any code may be too aggressive (many good prose posts include a code snippet). A threshold (e.g., >50% code by character count) is more reasonable but requires defining what counts as code.

5. **Token counting consistency.** Dolma reports in LLaMA tokens. Your downstream analysis likely uses word-level counts. Establish the conversion factor early and report both.
