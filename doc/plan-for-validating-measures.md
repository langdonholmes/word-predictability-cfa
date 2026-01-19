# Code Review: Phraseological Sophistication Indicators

## Quick Start: What You Need to Know

**Status**: The S_Phra factor fails because amod/advmod/dobj correlate at only r = 0.07-0.13 (need r > 0.3).

**Next Step**: Create diagnostic notebook `src/4-diagnostic-phraseological.ipynb` to generate visualizations and statistics (3-4 hours). This will confirm findings and determine which fixes to prioritize.

**Your Goals**:
- ✅ Keep trying for unified S_Phra factor
- ✅ Start with diagnostics before implementing fixes

**Most Likely Path**: After diagnostics → Apply frequency threshold + NPMI → Test if intercorrelations improve → If yes, re-run bifactor model. If no, add alternative measures (ACL ratio, RTTR).

---

## Executive Summary

**Primary Finding**: The phraseological sophistication factor (S_Phra) is non-functional because the three indicators (amod, advmod, dobj) have extremely weak intercorrelations (r = 0.07-0.13), far below the threshold needed for coherent factor formation (r > 0.3).

**Root Cause**: The MI calculation methodology is fundamentally flawed for measuring L2 phraseological sophistication due to:
1. **Inappropriate reference corpus** (Slim Pajama - heterogeneous LLM training data vs. native speaker corpus)
2. **No frequency thresholds** (90%+ of collocations occur ≤5 times, making PMI estimates unstable)
3. **Systematic missing value bias** (non-native collocations silently excluded from calculation)
4. **Construct invalidity** (negative MI values indicate learners use unconventional collocations, inverting the sophistication construct)

**Implementation Plan**: Start with comprehensive diagnostics (Phase 1), then apply targeted fixes (Phase 2), escalating to alternative measures (Phase 3) or reference corpus replacement (Phase 4) if needed. Goal is to improve intercorrelations to support unified S_Phra factor.

---

## Critical Issues Identified

### Issue 1: Inappropriate Reference Corpus ⚠️ CRITICAL

**Problem**: Using [Slim Pajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) (627B token LLM training corpus) instead of a balanced native speaker corpus like BNC.

**Evidence**:
- Slim Pajama composition: 52% web crawl, 5% code, 5% ArXiv papers, 3% StackExchange
- Paquot's methodology explicitly uses native speaker corpora (BNC) representing conventional English
- Different text types create inconsistent baselines across relation types:
  - Code/technical writing inflates advmod frequencies
  - Q&A forums affect dobj patterns
  - Web text is stylistically heterogeneous

**Impact**: MI values don't reflect "native-like sophistication" - they reflect alignment with a heterogeneous corpus designed for LLM training, not linguistic analysis.

---

### Issue 2: No Frequency Thresholds ⚠️ CRITICAL

**Problem**: No minimum frequency cutoff applied to collocations before computing MI.

**Evidence from data**:
```
amod:   67.8% of pairs occur once, 90.9% occur ≤5 times, 95.2% occur ≤10 times
advmod: 72.3% of pairs occur once, 92.4% occur ≤5 times, 96.1% occur ≤10 times
dobj:   69.1% of pairs occur once, 91.8% occur ≤5 times, 95.7% occur ≤10 times
```

**Code location**: [src/3-calculate-metrics-clc-fce.ipynb](src/3-calculate-metrics-clc-fce.ipynb) lines 128-149 (MiCalculator class)

```python
joint_count = self.dep_counts.get(pair, 0)
if joint_count == 0:
    continue  # Only excludes pairs with count=0, not low-frequency pairs
```

**Impact**: PMI is notoriously unstable for rare events. Computing MI for pairs seen once in 306M dependencies creates extreme variance and unreliable scores.

**Best practice**: Paquot and collocation literature typically require minimum frequency thresholds (5-10+ occurrences) for statistical reliability.

---

### Issue 3: Silent Missing Value Exclusion ⚠️ CRITICAL

**Problem**: Dependency pairs not found in reference corpus are completely excluded from mean calculation without documentation.

**Code location**: [src/3-calculate-metrics-clc-fce.ipynb](src/3-calculate-metrics-clc-fce.ipynb) lines 140-141

```python
if joint_count == 0:
    continue  # Silently skip this dependency
```

**Impact**:
- Novel, creative, or non-native collocations specific to learner language are dropped
- Mean is computed over biased sample of only corpus-attested pairs
- Lower-proficiency learners likely have more non-native collocations → more missing values → different sample sizes
- Undermines construct validity - we're measuring "how conventional are your conventional expressions" not "phraseological sophistication"

**Coverage appears good but hides the problem**:
- 99.2% of essays have ≥1 amod match
- 99.9% have ≥1 advmod match
- 99.8% have ≥1 dobj match

But within each essay, an unknown proportion of collocations are silently dropped.

---

### Issue 4: Negative MI Values Indicate Construct Inversion ⚠️ CRITICAL

**Problem**: All three indicators show predominantly negative MI values, opposite of Paquot's findings.

**Evidence**:
| Indicator | Mean MI | % Negative | % Positive |
|-----------|---------|------------|------------|
| amod      | -2.43   | 88.9%      | 11.1%      |
| advmod    | -5.39   | 99.3%      | 0.7%       |
| dobj      | -6.68   | 100.0%     | 0.0%       |

**Interpretation**: Learners are using **unconventional** collocations relative to Slim Pajama (words co-occur LESS than expected by chance).

**Expected from Paquot**: Positive MI indicating sophisticated, formulaic expressions that co-occur MORE than expected.

**Why this matters**:
- Higher MI (less negative) = fewer errors (r = -0.09 to -0.15)
- This suggests negative MI captures "non-nativeness" not "sophistication"
- The construct is fundamentally inverted

---

### Issue 5: Extremely Weak Intercorrelations ⚠️ CRITICAL

**Problem**: The three phraseological indicators don't correlate with each other.

**Evidence**:
```
Correlation Matrix:
           amod    advmod   dobj
amod      1.000    0.084   0.127
advmod    0.084    1.000   0.074
dobj      0.127    0.074   1.000
```

**Why this prevents factor formation**:
- For coherent factor: expect r = 0.3-0.7
- Observed: r = 0.07-0.13 (essentially uncorrelated)
- Shared variance among indicators is ~1% (r² ≈ 0.01)
- No common construct to extract

**Root cause**: Different syntactic relations have inherently different selectional restrictions in natural language:
- Verbs strongly constrain objects (eat food, *eat car) → very negative MI
- Adverbs constrain verbs moderately → moderately negative MI
- Adjectives combine more freely with nouns → less negative MI
- This creates a 4+ point spread in means that dominates any proficiency signal

---

### Issue 6: Model Mismatch Between Corpora ⚠️ MODERATE

**Problem**: CLC-FCE learner data parsed with `en_core_web_trf` (transformer), but Slim Pajama reference corpus parsed with `en_core_web_lg` (CNN).

**Evidence**:
- [src/1-spacy-clc-fce.ipynb](src/1-spacy-clc-fce.ipynb): Uses `en_core_web_trf` for learner corpus
- [src/1-spacy-slim-pajama.ipynb](src/1-spacy-slim-pajama.ipynb): Uses `en_core_web_lg` for reference corpus

**Impact**: Different parsers may label dependency relations differently, creating systematic bias in MI calculations (comparing apples to oranges).

---

### Issue 7: Error Classification - Minor Issues Only ✓ MOSTLY CORRECT

**Finding**: Error classification mapping is generally correct but has a few minor issues.

**Verified against CLC error codes**:
- Grammar category: Correctly includes AG*, AS, D*, F*, I* (except ID), M*, QL, R*, TV, U*, W, X, C[DN]Q
- Vocab category: Correctly includes CL, CE, L, ID
- Spelling category: Correctly includes S, SX

**Minor discrepancies found**:
1. Missing some compound codes that exist in the full CLC (e.g., AGA, AGD, AGN, AGQ, AGV under AG)
   - These are sub-codes; parent code 'G' is included, may be sufficient
2. Code 'S' used for both "Argument Structure" (AS) and "Spelling" (S)
   - Handled correctly: 'S' mapped to grammar, 'S' also mapped to spelling
   - Actually this is a collision - need to verify which 'S' appears in the data

**Recommendation**: This is not a priority issue - focus on the MI calculation problems first.

---

## Why S_Phra Factor Is Non-Functional

Your bifactor model showed:
- S_Phra variance at lower bound (0.100)
- Only amod loads significantly (0.316)
- advmod and dobj non-significant (-0.192, 0.144)

**Explanation from code review**:

1. **Minimal shared variance**: With intercorrelations of 0.07-0.13, there's almost no common variance for a specific factor to extract (~1% shared variance)

2. **General factor captures the weak signal**: The small proficiency-related variance that exists (r ≈ 0.10-0.15 with MTLD and errors) is absorbed by the general factor

3. **Between-relation variance dominates**: Systematic differences in MI baselines across relation types (4+ point spread) are as large as within-relation variation, creating a confound that masks proficiency signals

4. **Construct validity failure**: Negative MI values suggest these indicators measure "conventionality" or "nativeness" rather than "sophistication" as theorized by Paquot

---

## Recommendations

### Immediate Diagnostic Steps

1. **Generate diagnostic visualizations**:
   - Histograms of MI distributions by relation type
   - Scatter plots: MI vs error rates (should show negative correlation)
   - Box plots: MI distributions by proficiency level
   - Coverage plots: proportion of collocations matched per text

2. **Compute alternative metrics**:
   - Count of amod/advmod/dobj relations per text (before filtering)
   - Proportion of relations matched in reference corpus
   - Median MI (less sensitive to outliers than mean)
   - MI percentile ranks within relation type

3. **Correlate with proficiency proxies**:
   - MI vs MTLD (current: r = 0.09-0.17)
   - MI vs error rates (current: r = -0.09 to -0.15)
   - Compare high vs low proficiency groups

### Option 1: Fix Current MI Implementation (Recommended Short-term)

**Changes to make**:

1. **Apply frequency threshold**:
   ```python
   MIN_COUNT = 10  # Require ≥10 occurrences in reference corpus
   if joint_count < MIN_COUNT:
       continue
   ```

2. **Handle missing values explicitly**:
   ```python
   # Option A: Assign penalty value
   if joint_count == 0:
       mi = -2.0  # Fixed penalty for non-native collocations

   # Option B: Use normalized PMI (NPMI) which bounds [-1, +1]
   npmi = mi / -math.log2(p_xy)
   ```

3. **Track coverage**:
   ```python
   metrics[f'{rel}_count'] = total_relations
   metrics[f'{rel}_matched'] = len(mis)
   metrics[f'{rel}_coverage'] = len(mis) / total_relations if total_relations > 0 else 0
   ```

4. **Use within-relation percentile ranks**:
   - Compute percentile of text's MI relative to all texts for that relation
   - Removes baseline differences between relation types
   - Creates comparable sophistication scores

**Files to modify**:
- [src/3-calculate-metrics-clc-fce.ipynb](src/3-calculate-metrics-clc-fce.ipynb) - MiCalculator class (lines 128-149)

**Expected impact**: May improve intercorrelations somewhat, but fundamental issues remain due to inappropriate reference corpus.

---

### Option 2: Switch to Appropriate Reference Corpus (Recommended Medium-term)

**Replace Slim Pajama with**:
1. **British National Corpus (BNC)** - Paquot's choice
   - 100M words of balanced native speaker English
   - Written (90%): newspapers, academic, fiction, non-fiction
   - Spoken (10%): conversations, lectures
   - Publicly available

2. **Corpus of Contemporary American English (COCA)**
   - 1B words, balanced across genres
   - Updated regularly through 2019

**Implementation**:
1. Parse BNC/COCA with same spaCy model (`en_core_web_trf`)
2. Extract dependency bigrams identical to current pipeline
3. Rebuild MI calculator with new reference corpus
4. Recompute all metrics

**Files to create/modify**:
- New: `src/1-spacy-bnc.ipynb` - Parse BNC corpus
- New: `src/1b-extract-bnc-depgrams.ipynb` - Extract dependency bigrams
- Modify: [src/3-calculate-metrics-clc-fce.ipynb](src/3-calculate-metrics-clc-fce.ipynb) - Point to new reference corpus

**Expected impact**: Should produce positive MI values aligned with Paquot's construct, may improve intercorrelations.

---

### Option 3: Alternative Phraseological Measures (Recommended Long-term)

**Add or replace with**:

1. **Academic Collocation List (ACL) ratio** (Ackermann & Chen 2013):
   - Binary measure: proportion of collocations in ACL
   - More stable than PMI for short texts
   - Directly relevant to FCE academic writing context
   - List publicly available

2. **Phraseological diversity (RTTR)**:
   - Root type-token ratio of phraseological units
   - Paquot's second dimension alongside sophistication
   - May capture different aspect of competence

3. **Normalized PMI (NPMI)**:
   - Bounds values between -1 and +1
   - More stable across corpora
   - Formula: `NPMI = MI / -log₂(P(x,y))`

4. **Alternative association measures**:
   - Log-Dice: Less sensitive to frequency than PMI
   - Delta P: Probability-based, asymmetric
   - t-score: Balances frequency and association

**Implementation strategy**:
- Start with ACL ratio (simplest, most stable)
- Add NPMI alongside raw MI (easy modification)
- If these work, consider RTTR for diversity dimension

**Expected impact**: May create more coherent factor structure by using measures designed for L2 assessment rather than general collocation extraction.

---

### Option 4: Abandon Unified Phraseological Factor

**Rationale**: The linguistic evidence suggests different syntactic relations (amod, advmod, dobj) capture fundamentally different constructions with different selectional restrictions.

**Alternative model structure**:

```r
model_revised <- '
  # General factor
  G =~ MTLD + token_freq + lexical_density +
       error_grammar + error_vocab + error_spelling

  # Specific factors
  S_Lex =~ MTLD + token_freq + lexical_density
  S_Acc =~ error_grammar + error_vocab + error_spelling

  # Drop S_Phra - treat as separate indicators
  # Or: Create relation-specific factors if you have multiple indicators per relation
'
```

**Consider**:
- Include individual phraseological indicators (amod, advmod, dobj) as covariates rather than latent factor indicators
- Test whether any single indicator (advmod shows strongest signal: r=0.17 with MTLD) should be included
- Add alternative phraseological measures (ACL ratio, RTTR) as separate constructs

---

## Implementation Plan (Aligned with User Preferences)

### USER DIRECTION:
- **Start with diagnostics first, then decide** on implementation approach
- **Keep unified S_Phra factor as goal** - focus on improving intercorrelations

---

### Phase 1: Comprehensive Diagnostic Analysis ⭐ START HERE

**Goal**: Generate visualizations and statistics to confirm findings and guide next steps

**Create new notebook**: `src/4-diagnostic-phraseological.ipynb`

#### 1.1 Distribution Visualizations

Generate plots to understand current MI distributions:

```python
# For each relation type (amod, advmod, dobj):
- Histogram with kernel density overlay
- Q-Q plot to check normality
- Box plot with outliers highlighted
- Violin plot comparing original vs corrected texts

# Combined visualizations:
- Overlaid distributions for all three relations (shows baseline differences)
- Faceted histograms by proficiency quartile
```

#### 1.2 Coverage Statistics

Understand how many collocations are being matched vs dropped:

```python
# For each text:
- Count total relations extracted (before filtering)
- Count relations matched in reference corpus
- Proportion matched (coverage ratio)

# Summary statistics:
- Coverage by relation type
- Coverage by proficiency level
- Coverage by error rate quintiles

# Visualizations:
- Coverage distribution histograms
- Scatter: coverage vs proficiency proxies
```

#### 1.3 Correlation Analysis

Comprehensive correlation matrices with significance tests:

```python
# Matrices to generate:
1. Phraseological indicators only (amod, advmod, dobj)
2. All 9 current indicators
3. Phraseological vs proficiency proxies (MTLD, errors)
4. Partial correlations controlling for text length

# Visualizations:
- Heatmaps with correlation coefficients
- Scatter plot matrix for phraseological indicators
- Scatter plots: each MI vs MTLD, vs error rates
```

#### 1.4 Proficiency Signal Validation

Test whether MI indicators capture proficiency at all:

```python
# Split texts into proficiency groups using:
- Error rate quartiles (low errors = high proficiency)
- MTLD quartiles (high MTLD = high proficiency)
- mean_prob quartiles (high prob = high proficiency)

# Compare MI distributions across groups:
- Box plots by proficiency level
- T-tests between high/low groups
- Effect sizes (Cohen's d)

# Expected pattern if construct is valid:
- Higher proficiency → higher MI (less negative)
```

#### 1.5 Reference Corpus Frequency Analysis

Understand the low-frequency pair problem:

```python
# From depgrams.parquet:
- Distribution of pair frequencies (histogram)
- Cumulative distribution: % of pairs by frequency threshold
- For each relation type separately

# For each text's MI values:
- What was the median reference frequency of matched pairs?
- Correlation: reference frequency vs computed MI
- Are low-frequency pairs driving extreme MI values?
```

#### 1.6 Missing Value Analysis

Investigate the silent exclusion problem:

```python
# For each text:
- How many relations had count=0 (excluded)?
- Proportion of relations dropped
- Is proportion dropped correlated with proficiency?

# Expected pattern:
- Lower proficiency texts may have more non-native collocations
- These get excluded more often
- Creates systematic bias
```

#### 1.7 Diagnostic Report Generation

Create comprehensive markdown report with:

1. **Summary statistics tables**:
   - Distribution properties (mean, SD, skewness, kurtosis)
   - Correlation matrices
   - Coverage statistics
   - Proficiency group comparisons

2. **Visualizations** (save as PNG):
   - All plots from sections 1.1-1.6
   - Organized by analysis type

3. **Statistical tests**:
   - Significance of correlations
   - Group difference tests
   - Effect sizes

4. **Recommendations**:
   - Based on findings, which fixes to prioritize?
   - Is unified S_Phra factor feasible?
   - Which alternative measures to explore?

**Output files**:
- `src/4-diagnostic-phraseological.ipynb` - Analysis notebook
- `doc/diagnostic-report-phraseological.md` - Findings report
- `doc/figures/` - Saved visualizations

**Expected time**: 3-4 hours

---

### Phase 2: Targeted Fixes Based on Diagnostics

**Goal**: Implement improvements to maximize intercorrelations for unified S_Phra factor

**Wait for Phase 1 results before implementing**. Likely fixes to prioritize:

#### 2.1 Frequency Threshold (High Priority)

Modify MiCalculator in [src/3-calculate-metrics-clc-fce.ipynb](src/3-calculate-metrics-clc-fce.ipynb):

```python
MIN_COUNT = 10  # Set based on Phase 1 frequency analysis
if joint_count < MIN_COUNT:
    continue
```

**Rationale**: Eliminate unstable MI estimates from rare pairs

#### 2.2 Normalized PMI (High Priority)

Add NPMI alongside raw MI:

```python
# NPMI bounds values [-1, +1], more comparable across relations
if p_xy > 0:
    mi = math.log2(p_xy / (p_x * p_y))
    npmi = mi / -math.log2(p_xy)
    rel_mis[relation].append((mi, npmi))
```

**Rationale**: May reduce baseline differences between relation types

#### 2.3 Within-Relation Percentile Ranks (Medium Priority)

```python
# After collecting all MI values across all texts:
# Compute percentile rank of each text's MI within that relation type
# This removes baseline differences entirely

amod_percentile = percentileofscore(all_amod_values, text_amod_value)
```

**Rationale**: Directly addresses baseline difference problem

#### 2.4 Coverage Tracking (Medium Priority)

```python
# Track diagnostic info:
metrics[f'{rel}_total_count'] = count_of_all_relations
metrics[f'{rel}_matched_count'] = count_of_matched_relations
metrics[f'{rel}_coverage'] = matched / total
```

**Rationale**: Enables checking if coverage is confound

#### 2.5 Alternative Aggregation (Low Priority)

```python
# Compute multiple aggregation methods:
metrics[f'{rel}_mean'] = np.mean(mis)
metrics[f'{rel}_median'] = np.median(mis)  # Less sensitive to outliers
metrics[f'{rel}_trimmed_mean'] = trim_mean(mis, 0.1)  # Drop top/bottom 10%
```

**Rationale**: Test if mean is best aggregation for this construct

**Testing after Phase 2**:
1. Recompute all metrics with fixes
2. Generate new correlation matrices
3. Compare to Phase 1 baseline
4. If intercorrelations improve (r > 0.3), re-run bifactor model in R
5. If still weak (r < 0.3), proceed to Phase 3

**Expected time**: 2-3 hours coding, 2-3 hours testing

---

### Phase 3: Alternative Measures (If Phase 2 Insufficient)

**Goal**: Add theoretically-grounded measures that may better capture phraseological sophistication

**Priority order for unified S_Phra factor**:

#### 3.1 Normalized PMI (NPMI)
- Already covered in Phase 2.2

#### 3.2 Academic Collocation List (ACL) Ratio
- Proportion of dependency pairs appearing in ACL
- Requires obtaining ACL dataset
- More stable than PMI for short texts
- Binary measure: in list vs not in list

#### 3.3 Phraseological Diversity (RTTR)
- Root type-token ratio of phraseological units
- Paquot's second dimension
- May correlate better across relation types
- Captures range rather than sophistication

**Implementation approach**:
- Start with easiest (NPMI from Phase 2)
- If intercorrelations still weak, add ACL ratio
- RTTR as last resort

**Expected time**: 4-6 hours per measure

---

### Phase 4: Reference Corpus Replacement (If All Else Fails)

**Goal**: Switch to linguistically appropriate baseline (BNC/COCA)

**Only pursue if**:
- Phase 2 fixes don't improve intercorrelations
- Phase 3 alternatives don't work
- You want theoretically pure implementation regardless

**Steps**:
1. Obtain BNC corpus
2. Parse with `en_core_web_trf` (match learner data parser)
3. Extract dependency bigrams
4. Rebuild MI calculator
5. Recompute all MI metrics
6. Test intercorrelations with BNC baseline

**Expected time**: 1-2 days (mostly corpus processing)

---

### Decision Tree After Each Phase

```
Phase 1: Diagnostics
├─ Findings confirm issues? → Yes → Continue to Phase 2
└─ Issues not as severe? → Revise plan

Phase 2: Targeted Fixes
├─ Intercorrelations improve to r > 0.3? → Yes → Re-run bifactor model, DONE
├─ Some improvement but r < 0.3? → Continue to Phase 3
└─ No improvement? → Reconsider unified factor approach, consult user

Phase 3: Alternative Measures
├─ New measures intercorrelate r > 0.3? → Yes → Test in bifactor model
├─ Still weak? → Continue to Phase 4
└─ Mixed results? → Consider hybrid approach

Phase 4: Reference Corpus
└─ Final attempt with BNC/COCA baseline
```

---

## Correlation Matrices (From Diagnostic Analysis)

### Phraseological Indicators Only

```
           amod    advmod   dobj
amod      1.000    0.084   0.127
advmod    0.084    1.000   0.074
dobj      0.127    0.074   1.000
```

**Interpretation**: Essentially uncorrelated (max r = 0.127). Cannot form coherent factor.

---

### Full Nine-Indicator Matrix

```
                MTLD  token_freq  lex_dens   amod   advmod   dobj  err_gram  err_voc  err_spl
MTLD            1.00      -0.36      0.27   0.09     0.17   0.11     -0.32    -0.05    -0.16
token_freq     -0.36       1.00     -0.34  -0.06    -0.08  -0.06      0.18     0.05     0.10
lex_density     0.27      -0.34      1.00  -0.02     0.01   0.06     -0.12    -0.01    -0.05
amod            0.09      -0.06     -0.02   1.00     0.08   0.13     -0.11    -0.01    -0.04
advmod          0.17      -0.08      0.01   0.08     1.00   0.07     -0.09    -0.02    -0.08
dobj            0.11      -0.06      0.06   0.13     0.07   1.00     -0.15    -0.04    -0.11
error_grammar  -0.32       0.18     -0.12  -0.11    -0.09  -0.15      1.00     0.06     0.33
error_vocab    -0.05       0.05     -0.01  -0.01    -0.02  -0.04      0.06     1.00     0.05
error_spelling -0.16       0.10     -0.05  -0.04    -0.08  -0.11      0.33     0.05     1.00
```

**Key observations**:
1. **Lexical indicators correlate well** (r = 0.27-0.36): S_Lex makes sense
2. **Error indicators correlate moderately** (r = 0.33): S_Acc makes sense
3. **Phraseological indicators don't correlate** (r = 0.07-0.13): S_Phra fails
4. **Phraseological → Lexical correlations weak** (r = 0.09-0.17): Limited shared variance
5. **Phraseological → Error correlations weak** (r = -0.09 to -0.15): Some proficiency signal but very small

---

## Files Involved

### Primary Code File
- [src/3-calculate-metrics-clc-fce.ipynb](src/3-calculate-metrics-clc-fce.ipynb) - Main metrics calculation
  - Lines 128-149: MiCalculator class (CRITICAL ISSUES HERE)
  - Lines 167-231: Error classification (minor issues only)
  - Lines 233+: Metric extraction pipeline

### Supporting Files
- [src/1-spacy-clc-fce.ipynb](src/1-spacy-clc-fce.ipynb) - Learner corpus parsing (uses en_core_web_trf)
- [src/1-spacy-slim-pajama.ipynb](src/1-spacy-slim-pajama.ipynb) - Reference corpus parsing (uses en_core_web_lg - MISMATCH)
- [src/2-calculate-predictability-clc-fce.ipynb](src/2-calculate-predictability-clc-fce.ipynb) - Predictability metrics (not affected)

### Data Files
- [data/clc_fce_metrics_predictability_original.csv](data/clc_fce_metrics_predictability_original.csv) - Combined metrics (2,482 texts)
- [data/slim_pajama_lists/depgrams.parquet](data/slim_pajama_lists/depgrams.parquet) - Reference corpus (46M dependency bigrams)

---

## Expected Deliverables

Based on your handoff document requests:

1. ✅ **Diagnostic Report**: Issues found in MI calculation (THIS DOCUMENT)
2. ✅ **Correlation Analysis**: Matrices for phraseological and all indicators (INCLUDED ABOVE)
3. ⏳ **Distribution Analysis**: Visualizations pending (Phase 1)
4. ⏳ **Corrected Code**: Awaiting your direction on which option to pursue
5. ⏳ **New Measures**: Awaiting decision on ACL, RTTR, NPMI priority
6. ✅ **Recommendations**: Provided above (4 options + implementation plan)

---

## References Used

- [SlimPajama Dataset](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) - Reference corpus documentation
- Paquot, M. (2019). The phraseological dimension in interlanguage complexity research. *Second Language Research*, 35(1), 121-145.
- Vandeweerd, N., Housen, A., & Paquot, M. (2021). Applying phraseological complexity measures to L2 French. *International Journal of Learner Corpus Research*, 7(2), 197-229.
- Ackermann, K., & Chen, Y. H. (2013). Developing the Academic Collocation List (ACL)

---

## Technical Notes

### Why Negative MI Values?

MI formula: `MI = log₂(P(x,y) / (P(x) × P(y)))`

- If P(x,y) > P(x) × P(y): Words co-occur MORE than expected → Positive MI (association)
- If P(x,y) < P(x) × P(y): Words co-occur LESS than expected → Negative MI (dissociation)
- If P(x,y) = P(x) × P(y): Independent → MI = 0

**Your data shows**: Learner collocations co-occur LESS frequently in Slim Pajama than expected by chance.

**Paquot's data showed**: Advanced learner collocations co-occur MORE frequently in BNC (positive MI = formulaic, conventional expressions).

**The difference**: Reference corpus matters enormously.

### Parser Consistency Issue

- CLC-FCE parsed with `en_core_web_trf` (transformer, 95%+ accuracy)
- Slim Pajama parsed with `en_core_web_lg` (CNN, ~92% accuracy)
- Different parsers may assign different dependency labels to same construction
- This creates systematic bias when computing joint probabilities

**Solution**: Reparse one corpus with the other's model, or ensure both use same model.
