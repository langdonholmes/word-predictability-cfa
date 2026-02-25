# Research Notes: Word Predictability CFA Project

---

## Decision: Exclusion of Phraseological Sophistication Factor (January 19, 2026)

After correcting a calculation bug in the MI computation, we tested whether phraseological sophistication indicators (amod, advmod, dobj) form a coherent factor. Despite MI values shifting to predominantly positive (as theoretically expected), the indicators showed very weak intercorrelations and weak correlations with proficiency proxies.

**Intercorrelations (corrected MI values):**

| Pair | r | p |
|------|---|---|
| amod-advmod | 0.005 | 0.796 |
| amod-dobj | 0.083 | < 0.001 |
| advmod-dobj | 0.039 | 0.050 |
| **Mean** | **0.043** | |

**Correlations with exam score (CEFR level):**

| Indicator | r | p |
|-----------|---|---|
| amod | 0.045 | 0.027 |
| advmod | 0.048 | 0.019 |
| dobj | 0.082 | < 0.001 |

Recent work characterizes phraseological sophistication as multidimensional, encompassing distinct dimensions of association strength, register specificity, and frequency (Paquot & Naets, 2025). Different syntactic relations also have different selectional restrictions, which may explain why amod, advmod, and dobj do not cohere as a single factor. We therefore exclude S_Phra from the bifactor model.

**Final model:** General factor (G) with two specific factors: S_Lex (lexical sophistication) and S_Acc (accuracy).

### References

- Paquot, M., & Naets, H. (2025). Phraseological sophistication as a multidimensional construct. *International Journal of Learner Corpus Research*, *11*(1), 217-251. https://www.jbe-platform.com/content/journals/10.1075/ijlcr.23033.paq

- Paquot, M. (2019). The phraseological dimension in interlanguage complexity research. *Second Language Research*, *35*(1), 121-145. https://journals.sagepub.com/doi/10.1177/0267658317694221

- Vandeweerd, N., Housen, A., & Paquot, M. (2021). Applying phraseological complexity measures to L2 French: A partial replication study. *International Journal of Learner Corpus Research*, *7*(2), 197-229.

---
