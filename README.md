# Word Predictability and L2 Proficiency

This project investigates where **word predictability** -- a measure derived from masked language model surprisal -- fits within the structure of second language (L2) writing proficiency. Rather than imposing a theoretical factor structure (e.g., CAF), we use network psychometric methods to empirically discover how word predictability relates to established linguistic features across lexical, syntactic, phraseological, and discourse levels.

## Data

- **ELLIPSE Corpus** (public): 6,482 L2 essays with dimensional scores (Overall, Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions) and 44 writing prompts. Source: `ELLIPSE_Final_github.csv`.
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
| Mean surprisal (`mean_loss`) | Predictability | ModernBERT |
| Loss variance (`var_loss`) | Predictability | ModernBERT |

`mean_entropy` is computed but excluded from network analysis due to near-perfect correlation with `mean_loss` (r = 0.97).

## Pipeline

Run stages in order from the repo root. Stages 0-2 build the SlimPajama reference corpus (only needed once). Stages 3-4 process ELLIPSE essays.

```bash
# Stage 0-2: SlimPajama reference corpus (skip if already built)
python src/pipeline/0_download_slim_pajama.py
python src/pipeline/1_spacy_slim_pajama.py
python src/pipeline/2_collate_slim_pajama.py

# Stage 3: Calculate predictability (ModernBERT surprisal) for each essay
#   Input:  data/ELLIPSE_Final_github.csv
#   Output: data/ELLIPSE_Final_github_w_predictability.csv
python src/pipeline/3_calculate_predictability.py

# Stage 4: Calculate linguistic features and merge with scores
#   Input:  data/ELLIPSE_Final_github_w_predictability.csv
#   Output: data/ellipse_metrics.csv
python src/pipeline/4_calculate_metrics.py

# Network analysis (R/Quarto)
quarto render src/ch1_network/network-analysis.qmd
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
  ELLIPSE_Final_github.csv  # Public ELLIPSE dataset (6,482 essays, 44 prompts)
  slim_pajama_docbins/      # Reference corpus DocBins
  slim_pajama_lists/        # N-gram and dependency statistics
  *.csv                     # Computed metrics and scores
```

## Requirements

Python 3.10+ with PyTorch, Transformers, spaCy, and pandas. See `requirements.txt` for full dependencies. R with lavaan, bootnet, qgraph, and EGAnet for network analysis.
