# Handoff Summary: Code Review for L2 Writing Proficiency Metrics

## Research Context

**Researcher**: Langdon, PhD candidate at Vanderbilt University, working on dissertation research examining word predictability as a measure of language proficiency using transformer models like BERT.

**Primary Research Question**: Where does "word predictability" (operationalized as `mean_prob`, the average token probability from a BERT-style masked language model) fit within a model of second language writing proficiency?

**Dataset**: CLC-FCE (Cambridge Learner Corpus - First Certificate in English)
- Contains human error annotations
- External proficiency scores available for validation
- Sample size: ~2,400+ essays
- Proficiency levels: B2 to C2 on CEFR scale

**Key Variable Under Investigation**: `mean_prob` = average probability of observed tokens when each token is masked and predicted by a BERT-style LLM. Higher values indicate more predictable/conventional word choices.

---

## Current Model Status

### Working Bifactor Model

A bifactor model with acceptable fit (CFI = 0.952, TLI = 0.904, RMSEA = 0.037, SRMR = 0.028) has been established with three specific factors:

```r
model_bifactor <- '
  # General factor loads on all indicators
  G =~ MTLD + token_freq + lexical_density +
       amod + advmod + dobj +
       error_grammar + error_vocab + error_spelling
  
  # Specific factors capture residual covariance within groups
  S_Lex =~ MTLD + token_freq + lexical_density
  S_Phra =~ amod + advmod + dobj
  S_Acc =~ error_grammar + error_vocab + error_spelling
  
  # Orthogonality constraints
  G ~~ 0*S_Lex + 0*S_Phra + 0*S_Acc
  S_Lex ~~ 0*S_Phra + 0*S_Acc
  S_Phra ~~ 0*S_Acc
'
```

### Critical Problem: S_Phra Factor is Non-Functional

The phraseological sophistication specific factor (S_Phra) shows:
- Variance at lower bound (0.100)
- Only one significant loading (amod = 0.316)
- Non-significant loadings for advmod (-0.192) and dobj (0.144)

**This means phraseological variance is almost entirely captured by the general factor, with no meaningful residual covariance among the phraseological indicators.**

This is unexpected given Paquot's (2019) theoretical and empirical work showing phraseological sophistication as a distinct dimension that discriminates proficiency levels.

---

## Current Indicator Operationalizations

### Lexical Sophistication (S_Lex)
| Indicator | Description | Calculation Method | Expected Direction |
|-----------|-------------|-------------------|-------------------|
| MTLD | Measure of Textual Lexical Diversity | Standard MTLD algorithm | Higher = more proficient |
| token_freq | Average word frequency | Mean log frequency from reference corpus | Lower = more proficient |
| lexical_density | Content words / total words | Ratio calculation | Higher = more proficient |

### Phraseological Sophistication (S_Phra)
| Indicator | Description | Calculation Method | Expected Direction |
|-----------|-------------|-------------------|-------------------|
| amod | Adjectival modifier sophistication | Mean PMI of adj+noun dependencies | Higher = more proficient |
| advmod | Adverbial modifier sophistication | Mean PMI of adv+verb dependencies | Higher = more proficient |
| dobj | Direct object sophistication | Mean PMI of verb+dobj dependencies | Higher = more proficient |

**NOTE**: These ARE PMI-based measures (not raw counts), following Paquot's operationalization. The issue is not a misunderstanding of the construct—something else is causing the factor to collapse.

### Accuracy (S_Acc)
| Indicator | Description | Calculation Method | Expected Direction |
|-----------|-------------|-------------------|-------------------|
| error_grammar | Grammatical errors | Count from CLC-FCE annotations | Lower = more proficient |
| error_vocab | Vocabulary errors | Count from CLC-FCE annotations | Lower = more proficient |
| error_spelling | Spelling errors | Count from CLC-FCE annotations | Lower = more proficient |

**NOTE**: All variables are currently z-scored before modeling.

---

## Areas Requiring Code Review

### 1. PMI Calculation Pipeline

The phraseological sophistication measures need careful scrutiny:

**Questions to investigate:**
- What reference corpus is being used for PMI calculation?
- How are the PMI values being aggregated per text (mean, median, weighted)?
- Are low-frequency collocations being handled appropriately? (PMI is known to inflate scores for rare co-occurrences)
- Is there a minimum frequency threshold for including collocations?
- How are missing/zero PMI values handled when a collocation doesn't appear in the reference corpus?

**Paquot's methodology (for comparison):**
- Used a large L1 reference corpus (e.g., BNC or similar)
- Focused on specific syntactic relations extracted via dependency parsing
- Reported that verb+direct object relations showed strongest discrimination at advanced levels
- Used root type-token ratio (RTTR) for diversity alongside PMI for sophistication

**Potential issues:**
- If PMI is calculated from learner corpus itself rather than L1 reference corpus, values may not reflect "native-like" sophistication
- If aggregation includes many zero/missing values, means may be unreliable
- Dependency parser accuracy could affect which relations are captured

### 2. Dependency Parsing Quality

**Questions to investigate:**
- Which dependency parser is being used (spaCy, Stanza, CoreNLP)?
- What is the parser's accuracy on learner language (which may contain errors)?
- Are all three relation types (amod, advmod, dobj) being extracted correctly?
- Are there systematic parsing errors that might affect one relation type more than others?

**Potential issues:**
- Learner errors may cause parsing failures or misclassifications
- The parser may be trained on native text and perform poorly on L2 text
- Different parsers use different dependency relation labels

### 3. Normalization and Scaling

**Questions to investigate:**
- How are the raw PMI values distributed? (Highly skewed distributions may need transformation before z-scoring)
- Are there outliers that are distorting the measures?
- Is z-scoring appropriate, or should robust scaling be used?
- Are error counts being normalized by text length before z-scoring?

**Potential issues:**
- PMI can produce extreme values for rare collocations
- Error counts naturally correlate with text length
- Z-scoring assumes approximate normality

### 4. Text Preprocessing

**Questions to investigate:**
- How is tokenization handled?
- Are punctuation, numbers, proper nouns being handled consistently?
- Is lemmatization being applied before frequency/PMI lookups?
- How are multi-word expressions or compounds handled?

---

## Potential New Measures to Consider

Based on the literature review, several additional or alternative measures might strengthen the model:

### Alternative Phraseological Measures

1. **Academic Collocation List (ACL) Ratio**
   - Proportion of collocations appearing in Ackermann & Chen's (2013) Academic Collocation List
   - Paquot used this as an alternative operationalization of sophistication
   - May be more stable than PMI for smaller texts

2. **Phraseological Diversity (RTTR)**
   - Root type-token ratio of phraseological units
   - Paquot's second dimension of phraseological complexity
   - Could be added alongside sophistication measures

3. **Normalized PMI (NPMI)**
   - Bounds PMI between -1 and +1
   - More comparable across different corpus sizes
   - May reduce the instability of raw PMI scores

4. **Log-Dice or t-score**
   - Alternative association measures less sensitive to frequency
   - Used in some collocation extraction studies

### Additional Lexical Measures

5. **Word Range**
   - Number of different frequency bands represented
   - Captures vocabulary breadth differently than MTLD

6. **Academic Word List (AWL) Coverage**
   - Proportion of words from Coxhead's AWL
   - Relevant for the FCE academic writing context

### Syntactic Complexity Alternatives

Note: Syntactic complexity was dropped from the current model due to negative correlations with other factors. However, more targeted measures might work:

7. **Noun Phrase Complexity**
   - Mean length of noun phrases
   - Number of pre/post modifiers per NP
   - Recent research (2023) shows NP complexity is a strong predictor of writing quality

8. **Subordination Ratio**
   - Clauses per T-unit (rather than raw clause counts)
   - May avoid the multicollinearity issues seen with fluency measures

---

## Correlation Matrix Context

Before the model was fit, the following correlation patterns were observed (from earlier analysis):

- SynComp correlated **negatively** with other factors
- PhraSoph and Accuracy showed **negative** correlation
- This pattern is consistent with the complexity-accuracy trade-off hypothesis

**Request**: Please output correlation matrices for:
1. The three current phraseological indicators (amod, advmod, dobj)
2. All nine current indicators
3. Any new measures calculated

If amod, advmod, and dobj are not highly intercorrelated, this would explain why they don't form a coherent specific factor.

---

## Technical Environment

**Expected tools/languages:**
- Python (likely with spaCy or similar for NLP)
- R (for statistical modeling with lavaan)
- Possibly TAALES, TAASSC, or similar linguistic analysis tools

**Key packages to look for:**
- Dependency parsing: spaCy, Stanza, or CoreNLP
- PMI calculation: Custom code or existing NLP toolkit
- Lexical measures: lexicalrichness, TAALES, or custom
- Error extraction: Likely custom code using CLC-FCE XML annotations

---

## Specific Code Review Tasks

### Task 1: Audit PMI Calculation
- Locate the PMI calculation code
- Document the reference corpus being used
- Check for minimum frequency thresholds
- Verify aggregation method (mean, handling of missing values)
- Test on sample texts to verify reasonable outputs

### Task 2: Audit Dependency Extraction
- Identify the parser being used
- Check extraction logic for amod, advmod, dobj relations
- Run parser on sample texts with known structures to verify accuracy
- Check how parser handles learner errors

### Task 3: Examine Distributions
- Generate histograms/density plots for all indicators before z-scoring
- Identify outliers or extreme values
- Check for floor/ceiling effects
- Recommend transformations if needed

### Task 4: Compute Diagnostic Correlations
- Generate correlation matrix for the three phraseological indicators
- If correlations are low (<0.3), the indicators may not be measuring a coherent construct
- Compare correlations to those reported in Paquot's studies

### Task 5: Implement Alternative Measures
- If issues are found, implement corrected versions
- Consider adding NPMI alongside or instead of raw PMI
- Consider adding ACL ratio as a supplementary measure
- Consider adding phraseological diversity (RTTR) measures

---

## Expected Deliverables

1. **Diagnostic Report**: Summary of any issues found in metric calculations
2. **Correlation Analysis**: Matrices showing relationships among indicators
3. **Distribution Analysis**: Visualizations of indicator distributions
4. **Corrected Code** (if issues found): Updated calculation pipelines
5. **New Measures** (if appropriate): Implementation of additional indicators
6. **Recommendations**: Which measures to include in revised model testing

---

## Key References for Context

- Paquot, M. (2019). The phraseological dimension in interlanguage complexity research. *Second Language Research*, 35(1), 121-145.
- Paquot, M. (2018). Phraseological competence: A useful toolbox to delimitate CEFR levels in higher education? *Language Assessment Quarterly*, 15, 19-43.
- Vandeweerd, N., Housen, A., & Paquot, M. (2021). Applying phraseological complexity measures to L2 French. *International Journal of Learner Corpus Research*, 7(2), 197-229.
- Dunn, K. J., & McCray, G. (2020). The place of the bifactor model in confirmatory factor analysis investigations into construct dimensionality in language testing. *Frontiers in Psychology*, 11, 1357.
- Kyle, K., & Crossley, S. A. (2015). Automatically assessing lexical sophistication. *TESOL Quarterly*.

---

## Contact Notes

The researcher (Langdon) is highly proficient in Python, R, NLP, and statistical modeling. Technical explanations are appropriate. The goal is to identify whether calculation issues explain the non-functional S_Phra factor, and if not, to consider whether alternative operationalizations might create a more coherent factor structure.
