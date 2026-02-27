# Cluster Exploration Notes

## Summary

Delta vectors (e_pred - e_obs) from ModernBERT predictions on ELLIPSE
essays cluster primarily by observed token POS/lexeme — but at sufficient
granularity (200/leaf, 68 clusters), several clusters align with
recognizable L2 error patterns and show strong proficiency gradients. The
earlier 12-cluster solution (500/leaf) merged these patterns into
uninterpretable grab-bags. Finer-grained solutions (100/leaf, 185
clusters) further sharpen proficiency gradients, though with diminishing
returns on interpretability.

This is a **qualified positive result** for the vector-analogy hypothesis:
the delta space does encode error-relevant structure, but it is entangled
with lexical identity and only separable at fine cluster granularity.


## Parameter Sweep

| min_cluster_size | method | n_clusters | % clustered | notes |
|-----------------|--------|-----------|-------------|-------|
| 100 | leaf | 185 | 61.7% | strongest proficiency gradients; many redundant splits |
| 200 | leaf | 68 | 42.3% | **recommended**: best interpretability/manageability tradeoff |
| 300 | leaf | 37 | 37.0% | some interpretable clusters survive, but merges begin |
| 500 | leaf | 12 | 30.9% | too coarse; error patterns merged into POS grab-bags |
| 500 | eom | 9 | 65.3% | one giant catch-all (27k tokens), uninformative |
| 800 | leaf | 8 | 27.8% | too coarse |

**Key finding**: interpretability is non-monotonic in cluster count. The
12-cluster solution looked like a clean negative result (all clusters =
POS/lexeme). But at 68 clusters, several of those POS groups sub-divide
into linguistically meaningful categories. The critical difference is that
finer clustering separates, e.g., copula "is" in incoherent syntax
(cluster 25) from copula "is" in homophone contexts (cluster 24), rather
than lumping them into a single AUX cluster.


## Interpretable Clusters (200/leaf)

### Tier 1: Clear error types with strong proficiency gradients

**Cluster 25 (n=230) — Copula "is" in structurally incoherent syntax**
- POS: AUX(97%), DEP: ROOT(30%), ccomp(21%), conj(14%)
- Proficiency: **61.3% low, 14.3% high** (strongest skew of any cluster)
- Concordance: "in conkers is work Thomas Jefferson", "in many country is
  laws is alreay in", "secod is all, in life very important to l..."
- Model predictions vary widely (to, he, the, and) — no consistent expected
  form, just syntactic incoherence around the copula
- **Interpretation**: Low-proficiency writers inserting "is" as a default
  copula/connector in structurally broken sentences. This is a genuine L2
  syntactic pattern, not just a lexeme cluster.

**Cluster 22 (n=289) — "the"/"they"/"there" confusion**
- POS: DET(63%), PRON(27%)
- Proficiency: **57.4% low, 19.0% high**
- Concordance shows "the" used where model expected pronouns: "the help me
  where was my classroom", "the will be different", "the won't get any
  where in life"
- Also includes "ther"/"theze"/"thedays" misspellings
- **Interpretation**: DET-PRON confusion, a recognized L2 error pattern.
  The model consistently predicts "they"/"them" where the student wrote
  "the" — the delta vector captures a genuine determiner→pronoun mismatch.

**Cluster 55 (n=409) — Severe misspellings (out-of-vocabulary)**
- POS: NOUN(44%), PROPN(24%), VERB(19%), ADJ(11%)
- Mean surprisal: **8.40** (highest among interpretable clusters)
- Proficiency: **49.9% low, 25.7% high**
- Concordance: "fialling", "foward", "fustrated", "fcatorry", "gretest",
  "cabelble", "gulliblety", "ppeaper", "poblation"
- Tokens so misspelled that ModernBERT cannot map them to any subword —
  embeddings are essentially random, so deltas are large and idiosyncratic
- **Interpretation**: Severe orthographic errors. The cluster captures a
  real proficiency signal — low-proficiency writers produce more tokens that
  are completely opaque to the model. POS-heterogeneous because the
  misspellings span all content categories.

**Cluster 8 (n=336) — Bare infinitive "be" + "because" misspellings**
- POS: AUX(64%), ADV(16%), VERB(9%), NOUN(6%)
- Proficiency: **52.1% low, 17.9% high**
- Near-centroid: bare "be" in non-standard syntax ("be yourself help you",
  "the people be successfully"), mixed with "beacuse"/"beacause"
  misspellings
- **Interpretation**: Converges two patterns sharing the "be" subword: (1)
  bare infinitive overuse (L2 grammar), (2) "because" misspellings (L2
  orthography). Both are strongly low-proficiency-associated. The shared
  subword embedding drives their co-clustering.

**Cluster 29 (n=357) — Negation "not" in unexpected positions**
- POS: NOUN(37%), PART(37%), SCONJ(12%), ADV(11%)
- DEP: neg(37%), advmod(23%), dobj(17%)
- Proficiency: **49.0% low, 24.9% high**
- Concordance: "have success es have better idea not think bab ideas",
  "because not body kwon when will be need help", "not is easy because is
  more responsability"
- Model predictions at these positions are highly variable — the model did
  not expect negation here
- **Interpretation**: Negation misplacement. The observed "not" appears in
  syntactically unexpected positions. This is a genuine syntactic error
  pattern, not a lexeme artifact.

**Cluster 24 (n=317) — "were"/"where"/"are" homophone/form confusion**
- POS: AUX(94%)
- DEP: ccomp(25%), aux(15%), ROOT(15%), advcl(12%), auxpass(10%)
- Proficiency: **46.4% low, 20.5% high**
- Concordance shows two sub-patterns:
  - "were" used where model predicts "where" with high probability (>0.4):
    "stories were his friends have seen", "it's like a community were we
    help each other"
  - "are" in broken syntax: "the online classes are do not have benefits",
    "technology may have a negative effect on are limitation"
- **Interpretation**: Homophone/form confusion between were/where/are. The
  model's high-confidence "where" predictions at many of these positions
  confirm this is a genuine spelling/morphological error, not just random
  copula use.


### Tier 2: Lexeme clusters with proficiency signal

These clusters are organized by lexical identity but show meaningful
proficiency gradients, suggesting the specific word's usage pattern
differs across proficiency levels.

**Cluster 41 (n=383) — "they" as subject pronoun**
- POS: PRON(96%), DEP: nsubj(51%), poss(25%)
- Proficiency: **51.7% low, 19.8% high**
- Low-proficiency examples show pronoun+possessive confusion: "they
  dream", "they life", "they persons", "they failature is her past"
- Model predicts "their"/"the" at many of these positions
- **Interpretation**: Primarily a lexeme cluster, but the they→their
  confusion pattern within it is a real L2 error with proficiency signal.

**Cluster 43 (n=525) — "you/your" pronouns**
- POS: PRON(93%), DEP: nsubj(37%), poss(30%)
- Proficiency: **49.1% low, 23.8% high**
- Similar pattern to 41: low-proficiency examples show "your" where "the"
  expected ("your cell phone" → model predicts "their"), and "you" used
  as impersonal subject where model expects other forms
- **Interpretation**: Second-person pronoun overuse. Low-proficiency writers
  use "you/your" as a default impersonal pronoun more frequently.

**Cluster 4 (n=370) — "because" as subordinator**
- POS: SCONJ(97%), DEP: mark(90%)
- Proficiency: **51.9% low, 22.2% high**
- Concordance: "because" used in places where model predicts "and", ","
  (coordinating conjunctions). Low-proficiency writers over-use "because"
  as a general clause connector.
- **Interpretation**: Clause-linking overuse of "because". The model
  expects coordination; the student provides subordination.

**Cluster 18 (n=323) — Determiner "a" in unexpected contexts**
- POS: DET(73%)
- Proficiency: **52.0% low, 21.4% high**
- Mixed with misspellings: "aesy" (easy), "aportunate", "aferneen",
  "acomplish", "acommplish"
- **Interpretation**: Partly a determiner-misuse cluster, partly a catch-all
  for misspellings starting with "a". The misspelling component adds noise.

**Cluster 46 (n=300) — "do" in non-standard auxiliary positions**
- POS: VERB(66%), AUX(31%)
- Proficiency: **54.0% low, 22.0% high**
- "do" used as main verb where model expects other verbs: "do
  accomplished thingas", "do the best potential", "how can I do graduated?"
- **Interpretation**: "do" overuse as a default/support verb, characteristic
  of lower L2 proficiency.

**Cluster 9 (n=322) — Preposition "for" in non-standard contexts**
- POS: ADP(88%), DEP: prep(85%)
- Proficiency: **55.6% low, 20.5% high**
- "for" used where model expects "to", "and", "of": "technology for give
  to know his products", "they had for that resance", "for be a good
  students"
- **Interpretation**: Preposition substitution. "For" is overused by
  low-proficiency writers as a general-purpose preposition, possibly due to
  L1 transfer (Spanish "para"/"por").


### Tier 3: Non-interpretable clusters

**Cluster 66 (n=1808) — Anonymization placeholders**
- POS: NOUN(41%), PROPN(31%), VERB(12%), ADJ(11%)
- Mean surprisal: **9.33** (highest of any cluster)
- Dominated by Generic_Name, Generic_City, Generic_School tokens
- **Verdict**: Corpus artifact. These placeholders are out-of-distribution
  for ModernBERT.

**Cluster 49 (n=407) — Sentence-initial windowing artifact**
- POS: mixed (NOUN 23%, VERB 17%, SCONJ 14%)
- Model predicts [CLS] or [SEP] with >90% probability at nearly all tokens
- **Verdict**: Windowing artifact. These tokens appear at the start of the
  64-subword context window, so the model defaults to predicting boundary
  tokens.

**Cluster 51 (n=269) — Numerals**
- POS: NUM(92%)
- Proficiency: 32.7% low, **37.9% high** (slight high skew)
- Almost entirely "three" and "four" (essay prompt asks about 3-year vs
  4-year graduation)
- **Verdict**: Topic-specific lexeme cluster. The slight high-proficiency
  skew may reflect that higher-proficiency writers engage more with the
  specific numerical prompt details.

**Clusters 56, 64, 65** — Adjective clusters (amod, acomp positions). POS-
organized, proficiency-balanced, no interpretable error pattern.

**Clusters 57, 28, 58, 62** — Verb clusters (ROOT, advcl, xcomp
positions). Organized by syntactic position, not by error type.

**Clusters 2, 12, 14, 15** — Preposition clusters (ADP 96-97%). Multiple
preposition clusters that differ in which specific preposition (for, of,
in, to) but don't capture error types.

**Clusters 19, 37, 38** — Punctuation clusters. Organized by punctuation
type, no error signal.

**Clusters 27, 23, 26** — Auxiliary "have"/"would"/"can" clusters. Lexeme-
organized, no consistent error pattern.


## Effect of Further Cluster Increases (100/leaf, 185 clusters)

Going from 68 to 185 clusters has **real but diminishing returns**:

### What improves
1. **Stronger proficiency gradients**: The "they" pronoun cluster splits
   into sub-clusters, with the most low-skewed reaching **68.5% low /
   11.8% high** (vs. 51.7%/19.8% at 68 clusters). This is because the
   finer split separates the most syntactically broken "they" uses from
   more conventional ones.

2. **New interpretable sub-patterns emerge**:
   - A "his/him/he" cluster (n=153, 58.8% low) separates from the general
     pronoun pool, capturing pronoun gender/number confusion ("his life"
     where model expects "their", "he take more money" where model expects
     "they")
   - A "think" verb cluster (n=139, 59.0% low) isolates mental-state verb
     misuse in low-proficiency writing
   - A Spanish-influenced misspelling cluster (n=169, 58.6% low) emerges
     with "comunity", "imposible", "enthusiasms", "estudent" — evidence
     of L1 transfer

3. **More tokens get clustered**: 61.7% vs 42.3%, reducing noise

### What doesn't improve
1. **Many of the 185 clusters are redundant lexeme splits**: e.g., multiple
   clusters for different prepositions, different punctuation marks,
   different topic-specific nouns. These don't add interpretive value.

2. **Manual inspection doesn't scale**: Reviewing 185 clusters by hand is
   impractical. The interpretable clusters are a minority (~10-15 of 185).

3. **The core interpretable clusters from 200/leaf are preserved**:
   Cluster 25 (copula "is") appears identically in both solutions (same
   230 tokens, same 61.3% low). The interpretable structure is stable.

### Recommendation
Use **200/leaf (68 clusters)** as the primary analysis. Cite the 100/leaf
result as evidence that further splitting *does* sharpen gradients — this
is useful for arguing that the proficiency signal is real and not an
artifact of cluster boundaries. But the 68-cluster solution is more
practical to present and interpret.


## Proficiency Gradients: Real or Base-Rate?

The proficiency skews in the interpretable clusters are substantially
larger than what base rates would predict:

- **Overall base rate** among clustered tokens: ~43% low, ~29% mid, ~28%
  high (low-proficiency writers produce more high-surprisal tokens)
- **Cluster 25** (copula "is"): 61.3% low — 18pp above base rate
- **Cluster 22** (the/they confusion): 57.4% low — 14pp above base rate
- **Cluster 55** (severe misspellings): 49.9% low — 7pp above base rate
- **Cluster 51** (numerals): 32.7% low — 10pp *below* base rate

The contrast between clusters like 25 (heavily low-skewed) and 51
(slightly high-skewed) within the same overall distribution argues that the
proficiency gradients reflect genuine differences in error patterns, not
just base rates. If all clusters were equally affected by the base rate,
they would all skew similarly.

At 100/leaf, the "they" pronoun sub-cluster reaches **68.5% low** — 25pp
above base rate. This is difficult to explain as a base-rate artifact.


## Why Lexical Identity Still Dominates

The delta vector is delta = e_pred - e_obs. The observed word's embedding
(e_obs) contributes a large, consistent component to delta — all tokens
where the student wrote "they" share a common e_obs regardless of what
error was made. This component dominates the 768-dim delta space.

At coarse granularity (12 clusters), the clustering algorithm has no
choice but to organize by this dominant signal. At finer granularity (68+
clusters), the within-POS/lexeme variation in e_pred (which *does* carry
error-type information) becomes large enough relative to cluster size for
sub-structure to emerge.

This explains the non-monotonic interpretability: too few clusters →
POS grab-bags; just enough clusters → POS subdivides into error-relevant
categories; too many clusters → diminishing returns as even the error-
relevant sub-structure gets split arbitrarily.


## Suggestions for Future Analysis

### 1. Condition on observed POS, then cluster within POS
Since POS dominates the delta space, stratify tokens by observed POS tag
first, then cluster within each group. This directly addresses the lexical
identity confound and could reveal error-type sub-structure within, e.g.,
all DET tokens or all AUX tokens.

### 2. Project out lexical identity
Compute mean delta per observed lemma and subtract: delta_residual =
delta - mean_delta[lemma]. Cluster the residuals. If error-type information
exists beyond lexical identity, it should emerge.

### 3. Cluster prediction distributions instead of delta vectors
Cluster the model's softmax output P(w | context) at each masked position.
Two tokens with the same error type should have similar prediction profiles
regardless of what was actually written.

### 4. Use predicted-vs-observed POS cross-tabulation
Define error categories by predicted POS vs. observed POS (DET→PRON,
VERB→AUX, etc.) and cross-tabulate with proficiency. Bypasses the
embedding space entirely.


## Reproduction

```bash
PY=/home/jovyan/conda_envs/hf/bin/python

# UMAP is already cached; re-run only if needed:
# $PY concordance.py reduce

# Recommended parameterization:
$PY concordance.py cluster --min-cluster-size 200 --min-samples 10 --method leaf

# Inspect interpretable clusters:
$PY concordance.py show 25 --n 40  # copula "is" in incoherent syntax
$PY concordance.py show 22 --n 40  # the/they/there confusion
$PY concordance.py show 55 --n 40  # severe misspellings
$PY concordance.py show 8 --n 40   # bare infinitive "be" + "because"
$PY concordance.py show 29 --n 40  # negation misplacement
$PY concordance.py show 24 --n 40  # were/where/are confusion

# Finer-grained comparison:
$PY concordance.py cluster --min-cluster-size 100 --min-samples 10 --method leaf
$PY concordance.py show 87 --n 30  # "they" sub-cluster, 68.5% low
$PY concordance.py show 90 --n 30  # "his/him" gender confusion
$PY concordance.py show 183 --n 30 # L1 transfer misspellings
```
