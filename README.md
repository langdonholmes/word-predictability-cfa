# Word Predictability and L2 Proficiency

This project investigates where **word predictability** -- a measure derived from masked language model surprisal -- fits within the structure of second language (L2) writing proficiency. Rather than imposing a theoretical factor structure (e.g., CAF), we use network psychometric methods to empirically discover how word predictability relates to established linguistic features across lexical, syntactic, phraseological, and discourse levels.

## Data

- **ELLIPSE Corpus** (public): 6,482 L2 essays with dimensional scores (Overall, Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions) and 44 writing prompts. Source: `ELLIPSE_Final_github.csv`.
- **Dolma v1.7**: sampled into two ~1.05B-token reference corpora used for token frequency and dependency MI calculations:
  - **Corpus A (edited prose)**: wiki, pes2o, cc_news
  - **Corpus B (unedited prose)**: reddit, stackexchange, cc_blogs

## Features

Linguistic features are computed from spaCy parses and a Dolma reference corpus:

| Feature | Level | Source |
|---------|-------|--------|
| MTLD | Lexical | lexicalrichness |
| Lexical density | Lexical | spaCy POS |
| Log token frequency | Lexical | Dolma reference |
| Mean word length | Lexical | spaCy |
| Words per sentence | Syntactic | spaCy |
| Clauses per T-unit | Syntactic | spaCy |
| Modifiers per nominal | Syntactic | spaCy |
| Mean dependency distance | Syntactic | spaCy |
| amod MI | Phraseological | Dolma reference |
| dobj MI | Phraseological | Dolma reference |
| advmod MI | Phraseological | Dolma reference |
| Content word overlap | Cohesion | spaCy |
| Connective density | Cohesion | spaCy + word list |
| Mean surprisal (`mean_loss`) | Predictability | ModernBERT |
| Loss variance (`var_loss`) | Predictability | ModernBERT |

`mean_entropy` is computed but excluded from network analysis due to near-perfect correlation with `mean_loss` (r = 0.97).

## Pipeline

The pipeline has two tracks that share numeric prefixes. The Dolma reference-corpus track (the `*_dolma` scripts) only needs to be built once; the essay track (`3_calculate_predictability`, `4_calculate_metrics`) processes ELLIPSE. Run all from the repo root.

```bash
# Reference corpus: sample, preprocess, verify, dependency-parse, and
# collate Dolma into frequency tables. Stages 3-4 run once per corpus.
python src/pipeline/0_sample_dolma.py
python src/pipeline/1_preprocess_dolma.py
python src/pipeline/2_verify_dolma.py
python src/pipeline/3_spacy_dolma.py --corpus a    # then --corpus b
python src/pipeline/4_collate_dolma.py --corpus a  # then --corpus b

# Essay track — Stage 3: word predictability (ModernBERT surprisal)
#   Input:  data/ELLIPSE_Final_github.csv
#   Output: data/ELLIPSE_Final_github_w_predictability.csv
python src/pipeline/3_calculate_predictability.py

# Essay track — Stage 4: linguistic features merged with scores
#   Input:  data/ELLIPSE_Final_github_w_predictability.csv
#   Output: data/ellipse_metrics.csv
python src/pipeline/4_calculate_metrics.py

# Network analysis (R/Quarto)
quarto render src/2-nomological-network/network-analysis.qmd
```

The network analysis residualizes all features on writing prompt before estimation to remove systematic prompt-driven variance.

## Project Structure

```
src/
  features/                 # Linguistic feature calculators
    predictability.py       # Word predictability (ModernBERT surprisal + variance)
    lexical.py              # MTLD, lexical density, token frequency, word length
    syntactic.py            # WPS, clauses/T-unit, modifiers/nominal, dep distance
    phraseological.py       # Dependency MI (amod, dobj, advmod)
    cohesion.py             # Content word overlap, connective density, sentence sim
  util/
    paths.py                # Canonical path constants (DATA_DIR, DOLMA_DIR, etc.)
    process_docs.py         # Batch spaCy DocBin processing utilities
    collation.py            # N-gram / depgram counting and parquet writers
  pipeline/                 # Data processing scripts (run in order)
    dolma_config.py         # Dolma sources and per-corpus sampling targets
    dolma_stream.py         # Streaming Dolma reader/sampler
    0_sample_dolma.py       # Sample corpora A and B from Dolma v1.7
    1_preprocess_dolma.py   # Clean, length-filter, dedup -> jsonl.gz
    2_verify_dolma.py       # Token / PII / duplicate verification report
    3_spacy_dolma.py        # Dependency-parse to DocBins (stride parallelism)
    4_collate_dolma.py      # Frequency tables (n-grams + dependency bigrams)
    3_calculate_predictability.py  # ELLIPSE ModernBERT surprisal
    4_calculate_metrics.py         # ELLIPSE features merged with scores
    ingest_essays.py        # ELLIPSE essay ingestion -> DocBins
  1-predictability-benchmark/   # Predictability benchmark
  2-nomological-network/        # Network psychometric analysis (R/Quarto)
  3-vector-transformations/     # NMF / vector-transformation experiments
data/
  ELLIPSE_Final_github.csv  # Public ELLIPSE dataset (6,482 essays, 44 prompts)
  dolma/
    corpus_a/, corpus_b/    # Raw sampled source shards
    preprocessed/           # Cleaned jsonl.gz + summaries
    docbins/                # spaCy DocBins (corpus_a, corpus_b)
    frequency_tables/       # Collated n-gram / depgram parquet (stage 4 output)
    logs/                   # Sampling and verification reports
  ellipse_docbins/          # ELLIPSE essay DocBins
  *.csv                     # Computed metrics and scores
```

## Requirements

Python 3.10+ with PyTorch, Transformers, spaCy, and pandas. See `requirements.txt` for full dependencies. R with lavaan, bootnet, qgraph, and EGAnet for network analysis.
