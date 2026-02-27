# Integrated Gradients Pilot: Findings

**Date:** 2026-02-27
**Script:** `src/5-attribution/ig_pilot.py`
**Data:** `data/ig_pilot/`

## Setup

We computed integrated gradients (IG) on ModernBERT-base to decompose
token-level surprisal into per-context-token attribution scores for 1,002
high-surprisal tokens (334 per proficiency tertile) sampled from the ELLIPSE
pilot set. Each attribution answers: "how much did this context token
contribute to the model's surprisal at the target position?"

- **Model:** `answerdotai/ModernBERT-base` (masked LM, 768-dim)
- **Baseline:** [PAD] embeddings (context positions replaced with PAD;
  [CLS], [SEP], [MASK] unchanged)
- **Layer:** `model.model.embeddings.tok_embeddings` (raw embedding lookup)
- **IG steps:** 100 (Gauss-Legendre quadrature)
- **Window:** 64 subword tokens centered on target
- **Runtime:** 693s on GPU (0.69s/token)

## Diagnostics

| Criterion | Target | Observed | Verdict |
|-----------|--------|----------|---------|
| Convergence (|delta| < 10% of surprisal diff) | >90% of tokens | 21.4% | Fail |
| Top-10 concentration | >70% of total |attribution| | 54.6% mean | Below threshold |
| Baseline surprisal | Well above actual | 20.0 vs 6.6 nats | Pass |
| Interpretability | Manual inspection | Plausible | Pass |
| Profile differentiation | Visible tertile differences | Yes | Pass |

### Convergence

The completeness axiom (sum of attributions = output difference) holds
poorly: mean |delta| is 10.4 nats against a ~13.4 nat surprisal difference.
The likely cause is that the raw `tok_embeddings` layer is followed by
LayerNorm and Dropout before the transformer blocks. Interpolating between
PAD and actual embeddings produces intermediate points that LayerNorm
rescales nonlinearly, making the integration path rough. The 100-step
quadrature doesn't capture this well.

This affects the *absolute scale* of attributions but not necessarily their
*relative ranking*. The qualitative interpretability of individual tokens
(see below) suggests the rankings are still informative.

### Concentration

Top-10 context tokens capture ~55% of total |attribution| per target, with
~60 context tokens in each window. This is lower than expected but
consistent with bidirectional masked LMs, which attend broadly across the
full context. Function words (determiners, prepositions, punctuation) each
carry small but nonzero signal about the masked position, spreading
attribution across many tokens.

## Qualitative findings

Individual token inspections via `ig_pilot.py show` produce linguistically
plausible attributions:

- **"people"** (low prof, surprisal 4.26): top attributions from "%", "are",
  "failing", "peoples" -- semantically relevant context words
- **"new"** (low prof, surprisal 6.51): top attributions from "the" (-1),
  "change" (-3), "calender" (+1) -- local syntactic frame
- **"hover"** (low prof, surprisal 17.37): high attribution from "action"
  (-1) and "the" (+1) -- immediate collocational context

Negative attributions (reducing surprisal) dominate, as expected: context
tokens generally make the target *more* predictable relative to the
uninformative PAD baseline.

## Profile findings

Attribution profiles (POS x distance band, mean attribution per cell) show:

1. **Strong distance decay** across all POS categories and tertiles.
   Adjacent tokens (distance 1) carry 3-4x the attribution magnitude of
   distant tokens (8+). This is consistent and universal.

2. **POS asymmetries.** Content words (NOUN, VERB, ADJ) carry larger
   attributions than function words at short distances; function words
   (DET, ADP) have more uniform profiles across distances.

3. **Tertile differences** are visible but modest in the signed profiles.
   The low-minus-high difference matrix shows the largest effects for
   DET at distance 1 (-1.34) and OTHER/DET at distances 4-7 (+0.60).
   Absolute-value profiles show subtler differences; the overall magnitude
   structure is similar across tertiles.

## Decision

The pilot answers both motivating questions positively, with caveats:

1. **Are IG attributions interpretable?** Yes -- individual token maps make
   linguistic sense on manual inspection.
2. **Do profiles differ by proficiency?** Somewhat -- distance decay is
   the dominant signal; tertile differences exist but are modest in
   magnitude.

The convergence issue must be addressed before scaling to the full pipeline.
The profiles are worth investigating further, but the effect sizes suggest
that proficiency-linked attribution differences may be subtle.

**Recommendation:** Proceed to Chapter 3, but fix the attribution layer
first.

## Next steps

1. **Fix convergence.** Attribute to the embedding layer *output* (after
   LayerNorm) rather than the raw lookup table. This should produce a
   smoother interpolation path. If that's insufficient, try n_steps=300
   or switch to a Riemann sum method.

2. **Validate with known cases.** Hand-pick 10-20 tokens with clear
   grammatical errors (e.g., tense violations, agreement errors) and verify
   that IG highlights the expected context tokens (e.g., past-tense verb
   for a tense error). This is more convincing than random-sample spot
   checks.

3. **Scale up.** Once convergence is satisfactory, run on the full 55,998
   high-surprisal tokens. At 0.69s/token this is ~10.7 hours on GPU; can
   be parallelized across essay batches.

4. **Add attribution categories.** The pilot uses POS x distance (32 cells).
   The full pipeline should add dependency relation and morphological
   features. Frequency band and discourse connectors are lower priority.

5. **Statistical testing.** With the full dataset, fit per-essay profiles
   and test for tertile differences with permutation tests or mixed-effects
   models rather than visual inspection.
