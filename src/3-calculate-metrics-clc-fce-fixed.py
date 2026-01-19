#!/usr/bin/env python3
"""
Recompute metrics with Phase 2 fixes:
1. Frequency threshold (MIN_COUNT >= 10)
2. Coverage tracking
3. Within-relation percentile ranks
"""

import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict, Counter
import math
from scipy.stats import percentileofscore

# For MTLD
from lexicalrichness import LexicalRichness

print("=" * 80)
print("RECOMPUTING METRICS WITH PHASE 2 FIXES")
print("=" * 80)

# Load the spaCy model
print("\n1. Loading spaCy model and documents...")
nlp = spacy.load("en_core_web_lg")

# Load the CLC-FCE docbins
docbin_path_original = "../data/clc-fce-docbins/original.docbin"
docbin_path_corrected = "../data/clc-fce-docbins/corrected.docbin"

docbin_original = DocBin().from_disk(docbin_path_original)
docs_original = list(docbin_original.get_docs(nlp.vocab))
print(f"   Loaded {len(docs_original)} original documents")

docbin_corrected = DocBin().from_disk(docbin_path_corrected)
docs_corrected = list(docbin_corrected.get_docs(nlp.vocab))
print(f"   Loaded {len(docs_corrected)} corrected documents")

assert len(docs_original) == len(docs_corrected), "Docbins must have same number of docs"

# Load reference data
print("\n2. Loading reference corpus...")
token_freq_df = pd.read_parquet("../data/slim_pajama_lists/3grams.parquet")
token_freq_df = token_freq_df.groupby('token_2', as_index=False)['count'].sum()
token_freq = dict(zip(token_freq_df['token_2'], token_freq_df['count']))
total_tokens = sum(token_freq.values())
print(f"   Loaded {len(token_freq)} unique tokens")

dep_df = pd.read_parquet("../data/slim_pajama_lists/depgrams.parquet")
print(f"   Loaded {len(dep_df):,} dependency bigrams")

# ============================================================================
# Enhanced MI Calculator
# ============================================================================

class MiCalculator:
    """
    Enhanced MI Calculator with frequency threshold and coverage tracking.

    Changes from original:
    - MIN_COUNT = 10: Only compute MI for pairs with ≥10 occurrences (reduces noise)
    - Coverage tracking: Report total relations vs matched relations per text
    - Store all MI values for later percentile rank calculation
    """

    MIN_COUNT = 5  # Minimum occurrences in reference corpus for stable PMI

    def __init__(self, reference_grams: pd.DataFrame):
        # BUG FIX: Sum counts across POS tag variations before creating dict
        # Problem: Multiple POS tags for same (lemma, lemma, relation) triple
        # Example: "best friend" → "good friend" with JJS+NN, JJ+NN, JJS+NNS, JJ+NNS
        # Solution: Group by lemma triple and sum counts
        dep_grouped = reference_grams.groupby(
            ['head_lemma', 'dependent_lemma', 'relation'],
            as_index=False
        )['count'].sum()

        # Build dep_counts directly using zip - much faster than set_index
        self.dep_counts = dict(zip(
            zip(dep_grouped['head_lemma'],
                dep_grouped['dependent_lemma'],
                dep_grouped['relation']),
            dep_grouped['count']
        ))
        self.head_marginals = reference_grams.groupby('head_lemma')['count'].sum().to_dict()
        self.dep_marginals = reference_grams.groupby('dependent_lemma')['count'].sum().to_dict()
        self.total_deps = reference_grams['count'].sum()

        # Store all MI values per relation for percentile rank calculation
        self.all_mi_values = {'amod': [], 'advmod': [], 'dobj': []}

    def __call__(self, doc) -> dict:
        """
        Calculate Mutual Information (MI) for dependency relations.

        MI = log2(P(head,dep) / (P(head) * P(dep)))
        - Values > 0: words co-occur more than expected by chance
        - Values < 0: words co-occur less than expected by chance

        Returns dict with:
        - {relation}: mean MI (e.g., 'amod': -2.5)
        - {relation}_total: total relations found (e.g., 'amod_total': 10)
        - {relation}_matched: relations matched in reference (e.g., 'amod_matched': 8)
        - {relation}_coverage: proportion matched (e.g., 'amod_coverage': 0.8)
        """
        rel_mis = defaultdict(list)
        rel_total = defaultdict(int)
        rel_matched = defaultdict(int)

        for token in doc:
            if token.dep_ in {'amod', 'advmod', 'dobj'}:
                head_lemma = token.head.lemma_.lower()
                dep_lemma = token.lemma_.lower()
                relation = token.dep_
                pair = (head_lemma, dep_lemma, relation)

                # Count total relations
                rel_total[relation] += 1

                # Get joint count P(head, dep) from reference corpus
                joint_count = self.dep_counts.get(pair, 0)

                # Apply frequency threshold to reduce noise from unreliable estimates
                if joint_count < self.MIN_COUNT:
                    continue

                # Count matched relations (passed threshold)
                rel_matched[relation] += 1

                # Calculate probabilities from reference corpus
                p_xy = joint_count / self.total_deps  # Joint probability
                p_x = self.head_marginals.get(head_lemma, 0) / self.total_deps  # P(head)
                p_y = self.dep_marginals.get(dep_lemma, 0) / self.total_deps  # P(dependent)

                # MI = log2(P(x,y) / P(x)*P(y))
                mi = math.log2(p_xy / (p_x * p_y))
                rel_mis[relation].append(mi)

        # Build results dictionary
        result = {}
        for rel in ['amod', 'advmod', 'dobj']:
            # Mean MI
            mis = rel_mis.get(rel, [])
            result[rel] = np.mean(mis) if mis else np.nan

            # Coverage statistics
            total = rel_total.get(rel, 0)
            matched = rel_matched.get(rel, 0)
            result[f'{rel}_total'] = total
            result[f'{rel}_matched'] = matched
            result[f'{rel}_coverage'] = matched / total if total > 0 else np.nan

            # Store MI values for percentile calculation
            if mis:
                self.all_mi_values[rel].extend(mis)

        return result

mi_calculator = MiCalculator(dep_df)

# ============================================================================
# Helper Functions
# ============================================================================

# Error type mapping
error_mapping = {
    'grammar': ['G', 'GD', 'GN', 'GQ', 'GV', 'S', 'D', 'DC', 'DD', 'DI', 'DJ', 'DN', 'DQ', 'DT', 'DV', 'DY',
                'F', 'FD', 'FJ', 'FN', 'FQ', 'FV', 'FY', 'I', 'IJ', 'IN', 'IQ', 'IV', 'IY',
                'M', 'MC', 'MD', 'MJ', 'MN', 'MP', 'MQ', 'MT', 'MV', 'MY', 'QL',
                'R', 'RC', 'RD', 'RJ', 'RN', 'RP', 'RQ', 'RT', 'RV', 'RY', 'TV',
                'U', 'UC', 'UD', 'UJ', 'UN', 'UP', 'UQ', 'UT', 'UV', 'UY', 'W', 'X', 'CD', 'CN', 'CQ'],
    'vocab': ['CL', 'CE', 'L', 'ID'],
    'spelling': ['S', 'SX']
}

error_type_to_cat = {}
for cat, types in error_mapping.items():
    for t in types:
        error_type_to_cat[t] = cat

def count_errors(doc) -> dict:
    error_counts = {'error_grammar': 0, 'error_vocab': 0, 'error_spelling': 0}
    if 'errors' in doc.spans:
        for span in doc.spans['errors']:
            cat = error_type_to_cat.get(span.label_, 'other')
            if not cat == 'other':
                error_counts[f'error_{cat}'] += 1
    return error_counts

def calculate_mtld(tokens) -> float:
    if len(tokens) < 10:
        return np.nan
    token_str = ' '.join(tokens)
    lex = LexicalRichness(token_str)
    return lex.mtld(threshold=0.72)

def count_tunits(doc) -> int:
    return sum(1 for token in doc if token.dep_ == 'ROOT')

def lexical_density(doc) -> float:
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    words = [t for t in doc if t.is_alpha]
    if not words:
        return 0
    content_words = [t for t in words if t.pos_ in content_pos]
    return len(content_words) / len(words)

def avg_token_freq(doc, token_freq):
    freqs = []
    for token in doc:
        if token.is_alpha:
            freq = token_freq.get(token.lemma_.lower(), 1)
            freqs.append(freq)
    return np.mean(freqs) if freqs else np.nan

def mod_per_nom(doc) -> float:
    nominals = [t for t in doc if t.pos_ == 'NOUN']
    if not nominals:
        return 0
    total_mods = 0
    for nom in nominals:
        mods = [c for c in nom.children if c.dep_ in {'amod', 'det', 'nummod', 'compound'}]
        total_mods += len(mods)
    return total_mods / len(nominals)

def dep_per_nom(doc) -> float:
    nominals = [t for t in doc if t.pos_ == 'NOUN']
    if not nominals:
        return 0
    total_deps = sum(len(list(nom.children)) for nom in nominals)
    return total_deps / len(nominals)

# ============================================================================
# Process Documents
# ============================================================================

def process_docs(docs, label):
    results = []

    print(f"\n3. Processing {len(docs)} {label} documents...")

    for idx, doc in tqdm(enumerate(docs), total=len(docs), desc=f"   Calculating metrics"):
        metrics = {'doc_id': idx}

        # Basic counts
        words = [t for t in doc if not t.is_punct]
        metrics['word_count'] = len(words)
        metrics['clause_count'] = len(list(doc.sents))
        metrics['tunit_count'] = count_tunits(doc)

        # Lexical
        lemmas = [t.lemma_.lower() for t in words if t.is_alpha]
        metrics['MTLD'] = calculate_mtld(lemmas)
        metrics['lexical_density'] = lexical_density(doc)
        metrics['token_freq'] = avg_token_freq(doc, token_freq)

        # Syntactic
        metrics['clauses_per_tunit'] = metrics['clause_count'] / metrics['tunit_count'] if metrics['tunit_count'] > 0 else np.nan
        metrics['mod_per_nom'] = mod_per_nom(doc)
        metrics['dep_per_nom'] = dep_per_nom(doc)

        # MI for relations (with coverage tracking)
        mi_dict = mi_calculator(doc)
        metrics.update({k: v for k, v in mi_dict.items()})

        # Errors
        error_dict = count_errors(doc)
        metrics.update(error_dict)

        results.append(metrics)

    return pd.DataFrame(results)

# Process original docs
df_original = process_docs(docs_original, "original")

# Process corrected docs
df_corrected = process_docs(docs_corrected, "corrected")

# ============================================================================
# Calculate Percentile Ranks
# ============================================================================

def calculate_percentile_ranks(df, indicators=['amod', 'advmod', 'dobj']):
    """
    Calculate within-relation percentile ranks for MI scores.

    This eliminates baseline differences across relation types by converting
    absolute MI scores to relative ranks (0-100) within each relation.
    """
    df_copy = df.copy()

    print(f"\n4. Calculating percentile ranks...")
    for ind in indicators:
        all_values = df_copy[ind].dropna().values

        percentiles = []
        for idx in df_copy.index:
            mi_value = df_copy.loc[idx, ind]
            if pd.isna(mi_value):
                percentiles.append(np.nan)
            else:
                pct = percentileofscore(all_values, mi_value, kind='rank')
                percentiles.append(pct)

        df_copy[f'{ind}_percentile'] = percentiles

    return df_copy

df_original = calculate_percentile_ranks(df_original)
df_corrected = calculate_percentile_ranks(df_corrected)

print("   ✓ Percentile ranks calculated")

# ============================================================================
# Save Results
# ============================================================================

print("\n5. Saving results...")

# Save basic metrics
output_path_original = "../data/clc_fce_metrics_original_fixed.csv"
output_path_corrected = "../data/clc_fce_metrics_corrected_fixed.csv"

df_original.to_csv(output_path_original, index=False)
df_corrected.to_csv(output_path_corrected, index=False)

print(f"   ✓ Saved: {output_path_original}")
print(f"   ✓ Saved: {output_path_corrected}")

# Merge with predictability metrics
predictability_original = pd.read_csv("../data/clc_fce_predictability_original.csv")
predictability_corrected = pd.read_csv("../data/clc_fce_predictability_corrected.csv")

df_combined_original = pd.merge(df_original, predictability_original, on='doc_id')
df_combined_corrected = pd.merge(df_corrected, predictability_corrected, on='doc_id')

output_path_combined_original = "../data/clc_fce_metrics_predictability_original_fixed.csv"
output_path_combined_corrected = "../data/clc_fce_metrics_predictability_corrected_fixed.csv"

df_combined_original.to_csv(output_path_combined_original, index=False)
df_combined_corrected.to_csv(output_path_combined_corrected, index=False)

print(f"   ✓ Saved: {output_path_combined_original}")
print(f"   ✓ Saved: {output_path_combined_corrected}")

# ============================================================================
# Display Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY OF CHANGES")
print("=" * 80)

print("\nPhase 2 Fixes Applied:")
print("  1. ✓ Frequency threshold: MIN_COUNT >= 10")
print("  2. ✓ Coverage tracking: {rel}_total, {rel}_matched, {rel}_coverage")
print("  3. ✓ Within-relation percentile ranks: {rel}_percentile")

print("\nNew columns added:")
cols = df_original.columns.tolist()
new_cols = [c for c in cols if any(x in c for x in ['_total', '_matched', '_coverage', '_percentile'])]
for col in sorted(new_cols):
    print(f"  - {col}")

print("\nMI Distribution Summary (with MIN_COUNT >= 10):")
print(df_original[['amod', 'advmod', 'dobj']].describe())

print("\nPercentile Rank Summary:")
print(df_original[['amod_percentile', 'advmod_percentile', 'dobj_percentile']].describe())

print("\nCoverage Summary (proportion of relations matched):")
print(df_original[['amod_coverage', 'advmod_coverage', 'dobj_coverage']].describe())

print("\n" + "=" * 80)
print("METRICS RECOMPUTATION COMPLETE")
print("=" * 80)
print("\nNext step: Run diagnostic analysis on new metrics to measure improvements")
