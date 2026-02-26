# Word Predictability and L2 Proficiency

This project investigates where **word predictability** -- a measure derived from masked language model surprisal -- fits within the structure of second language (L2) writing proficiency. Rather than imposing a theoretical factor structure (e.g., CAF), we use network psychometric methods to empirically discover how word predictability relates to established linguistic features across lexical, syntactic, phraseological, and discourse levels.

## Data

- **ELLIPSE Corpus**: ~136K L2 essays with dimensional rater scores (Overall, Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions)
- **SlimPajama**: ~458K documents used as a reference corpus for token frequency and dependency MI calculations

## Features

Linguistic features are computed from spaCy parses and a SlimPajama reference corpus:

| Feature | Level | Source |
|---------|-------|--------|
| MTLD | Lexical | lexicalrichness |
| Lexical density | Lexical | spaCy POS |
| Log token frequency | Lexical | SlimPajama reference |
| Mean word length | Lexical | spaCy |
| Words per sentence | Syntactic | spaCy |
| Clauses per T-unit | Syntactic | spaCy |
| Modifiers per nominal | Syntactic | spaCy |
| Mean dependency distance | Syntactic | spaCy |
| amod MI | Phraseological | SlimPajama reference |
| dobj MI | Phraseological | SlimPajama reference |
| advmod MI | Phraseological | SlimPajama reference |
| Content word overlap | Cohesion | spaCy |
| Connective density | Cohesion | spaCy + word list |
| Mean surprisal | Predictability | ModernBERT |
| Mean entropy | Predictability | ModernBERT |

## Project Structure

```
src/
  features/                 # Linguistic feature calculators
    predictability.py       # Word predictability (ModernBERT surprisal)
    lexical.py              # MTLD, lexical density, token frequency, word length
    syntactic.py            # WPS, clauses/T-unit, modifiers/nominal, dep distance
    phraseological.py       # Dependency MI (amod, dobj, advmod)
    cohesion.py             # Content word overlap, connective density, sentence sim
  util/
    paths.py                # Canonical path constants (DATA_DIR, FIG_DIR, etc.)
    process_docs.py         # Batch spaCy processing utilities
  pipeline/                 # Data processing scripts (run in order)
    0_download_slim_pajama.py
    1_spacy_slim_pajama.py
    2_collate_slim_pajama.py
    3_calculate_predictability.py
    4_calculate_metrics.py
  ch1_network/              # Network psychometric analysis (R/Quarto)
  ch2_predictability/       # (planned)
  ch3_invariance/           # (planned)
  ch4_xai/                  # (planned)
data/
  slim_pajama_docbins/      # Reference corpus DocBins
  slim_pajama_lists/        # N-gram and dependency statistics
  *.csv                     # Computed metrics and scores
```

## Requirements

Python 3.10+ with PyTorch, Transformers, spaCy, and pandas. See `requirements.txt` for full dependencies. R with lavaan, bootnet, qgraph, and EGAnet for network analysis.
