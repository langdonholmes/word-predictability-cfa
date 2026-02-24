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
  predictor.py              # Word predictability calculation (ModernBERT)
  process_docs.py           # Batch spaCy processing utilities
  0-download-slim-pajama    # Download reference corpus
  1-spacy-*                 # Parse texts to spaCy DocBin format
  2-calculate-*             # Calculate predictability scores
  3-calculate-metrics-*     # Calculate linguistic features
  sem/                      # Structural equation / network modeling (R/Quarto)
data/
  slim_pajama_docbins/      # Reference corpus DocBins
  slim_pajama_lists/        # N-gram and dependency statistics
  *.csv                     # Computed metrics and scores
```

## Requirements

Python 3.10+ with PyTorch, Transformers, spaCy, and pandas. See `requirements.txt` for full dependencies. R with lavaan, bootnet, qgraph, and EGAnet for network analysis.
