"""Integrated Gradients attribution profiling for nominal tokens across proficiency.

Triangulates the AttnLRP nominal attribution results (nominal_attribution.py)
with Captum's Integrated Gradients. Key methodological differences:
  - Targets -log_prob (surprisal) with explicit baselines, not raw logits
  - Uses interpolation paths (100 steps) rather than a single backward pass
  - No LXT monkey-patching needed (Captum handles gradient integration)

Subcommands:
    compute  — Sample nominal tokens, run IG, save attributions
    analyze  — Compare attribution profiles by proficiency tertile
    show     — Inspect individual token attribution maps

Example workflow:
    python nominal_ig.py compute --device cuda
    python nominal_ig.py analyze
    python nominal_ig.py show

Source data:
    data/lrp_pilot/token_pos_cache.parquet — POS/dep_rel lookup (from nominal_attribution.py)
    data/ELLIPSE_Final_github.csv — essay texts + scores
    data/ELLIPSE_token_predictability.parquet — token-level surprisal

Outputs (in data/lrp_pilot/):
    nominal_ig_attrs.parquet  — one row per (target, context) pair
    nominal_ig_meta.parquet   — one row per target token
    nominal_ig_params.json    — hyperparameters and runtime stats
    fig/nominal_ig.pdf        — multi-panel comparison figure
"""

import sys
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.paths import DATA_DIR, FIG_DIR

from lrp_pilot import (
    get_token_alignment,
    get_pos_coarse,
    get_distance_band,
    POS_COARSE_MAP,
    POS_ORDER,
    BAND_ORDER,
)

ELLIPSE_PATH = DATA_DIR / "ELLIPSE_Final_github.csv"
FULL_TOKEN_PATH = DATA_DIR / "ELLIPSE_token_predictability.parquet"
DOCBIN_PATH = DATA_DIR / "ellipse_docbins" / "processed_docs_00.spacy"
OUTPUT_DIR = DATA_DIR / "lrp_pilot"
POS_CACHE_PATH = OUTPUT_DIR / "token_pos_cache.parquet"

MODELS = {
    "bert": "bert-base-uncased",
    "modernbert": "answerdotai/ModernBERT-base",
}


# ── POS cache ────────────────────────────────────────────────────────────

def build_pos_cache():
    """Build (essay_id, spacy_idx) → (pos_tag, dep_rel) cache from DocBin."""
    import spacy
    from spacy.tokens import DocBin

    print(f"Building POS cache from {DOCBIN_PATH}...")
    nlp = spacy.load("en_core_web_lg")
    doc_bin = DocBin().from_disk(DOCBIN_PATH)
    docs = list(doc_bin.get_docs(nlp.vocab))
    print(f"  Loaded {len(docs)} docs from DocBin")

    essays_df = pd.read_csv(ELLIPSE_PATH)

    def alnum_key(text):
        return re.sub(r"[^a-zA-Z0-9]", "", text)[:80]

    essay_lookup = {}
    for _, row in essays_df.iterrows():
        key = alnum_key(row["full_text"])
        essay_lookup[key] = row["text_id_kaggle"]

    pos_rows = []
    n_matched = 0
    for doc in docs:
        key = alnum_key(doc.text)
        if key not in essay_lookup:
            continue
        eid = essay_lookup[key]
        n_matched += 1
        for tok in doc:
            pos_rows.append({
                "text_id_kaggle": eid,
                "spacy_idx": tok.i,
                "pos_tag": tok.pos_,
                "dep_rel": tok.dep_,
            })

    print(f"  Matched {n_matched}/{len(essays_df)} essays, "
          f"{len(pos_rows)} tokens")

    pos_df = pd.DataFrame(pos_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pos_df.to_parquet(POS_CACHE_PATH, index=False)
    print(f"  Cached → {POS_CACHE_PATH}")
    return pos_df


def load_pos_cache():
    """Load POS cache, building it if needed."""
    if POS_CACHE_PATH.exists():
        print(f"Loading POS cache from {POS_CACHE_PATH}")
        return pd.read_parquet(POS_CACHE_PATH)
    return build_pos_cache()


# ── compute ──────────────────────────────────────────────────────────────

def cmd_compute(args):
    """Sample nominal tokens and compute IG attributions."""
    import spacy
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
    from captum.attr import LayerIntegratedGradients
    from features.predictability import get_centered_window

    # Step 1: Load POS cache
    pos_df = load_pos_cache()

    # Step 2: Load token predictability, merge POS
    print(f"\nLoading token data from {FULL_TOKEN_PATH}...")
    full_tokens = pd.read_parquet(FULL_TOKEN_PATH)
    full_tokens = full_tokens.rename(columns={
        "text_id_kaggle": "essay_id",
        "text": "token_text",
        "mean_loss": "surprisal",
    })

    full_tokens = full_tokens.merge(
        pos_df.rename(columns={"text_id_kaggle": "essay_id"}),
        on=["essay_id", "spacy_idx"],
        how="inner",
    )
    print(f"  After POS merge: {len(full_tokens)} tokens")

    # Step 3: Filter to nominals
    nominals = full_tokens[full_tokens["pos_tag"].isin(["NOUN", "PROPN"])].copy()
    print(f"  Nominal tokens (NOUN/PROPN): {len(nominals)}")

    # Join with essay scores for proficiency tertiles
    essays_df = pd.read_csv(ELLIPSE_PATH)
    t_low, t_high = essays_df["Overall"].quantile([1/3, 2/3]).values
    essays_df["prof_tertile"] = pd.cut(
        essays_df["Overall"],
        bins=[0, t_low, t_high, 6],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    nominals = nominals.merge(
        essays_df[["text_id_kaggle", "prof_tertile"]].rename(
            columns={"text_id_kaggle": "essay_id"}
        ),
        on="essay_id",
        how="inner",
    )

    # Exclude boundary tokens
    essay_lengths = nominals.groupby("essay_id")["spacy_idx"].transform("max")
    nominals = nominals[
        (nominals["spacy_idx"] > 0) & (nominals["spacy_idx"] < essay_lengths)
    ]
    print(f"  After excluding boundary tokens: {len(nominals)}")
    print(f"  Tertile distribution:\n{nominals['prof_tertile'].value_counts().to_string()}")

    # Step 4: Sample
    rng = np.random.default_rng(args.seed)
    samples = []
    for tertile in ["low", "mid", "high"]:
        tdf = nominals[nominals["prof_tertile"] == tertile]
        n = min(args.n_per_tertile, len(tdf))
        idx = rng.choice(len(tdf), size=n, replace=False)
        samples.append(tdf.iloc[idx])
    sample_df = pd.concat(samples).reset_index(drop=True)
    print(f"\nSampled {len(sample_df)} nominal tokens: "
          f"{sample_df['prof_tertile'].value_counts().to_dict()}")

    # Step 5: Load model (no LXT patching — IG uses Captum)
    model_key = args.model
    model_name = MODELS[model_key]

    print(f"\nLoading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_lg")

    print("Loading essays...")
    essay_texts = dict(zip(essays_df["text_id_kaggle"], essays_df["full_text"]))

    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    # Embedding layer for IG
    if model_key == "bert":
        embed_layer = model.bert.embeddings.word_embeddings
    else:
        embed_layer = model.model.embeddings.tok_embeddings

    # Baseline fill token
    fill_id = pad_id if args.baseline == "pad" else mask_id

    def forward_fn(input_ids, attention_mask, mask_pos, target_id):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs.logits[:, mask_pos, :], dim=-1)
        return -log_probs[:, target_id]

    lig = LayerIntegratedGradients(forward_fn, embed_layer)

    essay_cache = {}

    def get_essay_data(essay_id):
        if essay_id not in essay_cache:
            text = essay_texts[essay_id]
            doc = nlp(text)
            token_map, trf_tok_ids = get_token_alignment(tokenizer, doc)
            essay_cache[essay_id] = (doc, token_map, trf_tok_ids)
        return essay_cache[essay_id]

    # IG attribution loop
    print(f"Baseline: {args.baseline} (n_steps={args.n_steps})")

    attr_rows = []
    meta_rows = []
    n_skipped = 0

    t0 = time.time()
    n_total = len(sample_df)
    print(f"\nComputing IG for {n_total} nominal tokens...")

    for i, (_, row) in enumerate(sample_df.iterrows()):
        essay_id = row["essay_id"]
        spacy_idx = int(row["spacy_idx"])

        try:
            doc, token_map, trf_tok_ids = get_essay_data(essay_id)
        except KeyError:
            n_skipped += 1
            continue

        if spacy_idx not in token_map:
            n_skipped += 1
            continue

        subword_start, subword_end = token_map[spacy_idx]
        seq_len = len(trf_tok_ids)
        effective_window = min(args.window_size, seq_len)

        win_start, win_end = get_centered_window(
            seq_len, subword_start, effective_window
        )
        window_ids = trf_tok_ids[win_start:win_end].copy()
        window_len = win_end - win_start

        tok_start_w = max(0, subword_start - win_start)
        tok_end_w = min(window_len, subword_end - win_start)
        if tok_end_w <= tok_start_w:
            n_skipped += 1
            continue

        # Mask all subwords of target token
        window_ids[tok_start_w:tok_end_w] = mask_id

        # Full sequence: [CLS] + window + [SEP]
        full_ids = np.concatenate([[cls_id], window_ids, [sep_id]])
        first_mask_pos = tok_start_w + 1
        actual_first_id = int(trf_tok_ids[subword_start])
        mask_positions = np.arange(tok_start_w, tok_end_w) + 1

        input_t = torch.from_numpy(
            full_ids.astype(np.int64)
        ).unsqueeze(0).to(args.device)
        attn_mask = torch.ones_like(input_t)

        # Baseline: fill with pad/mask, keep CLS/SEP/target-mask positions
        baseline_ids = np.full_like(full_ids, fill_id)
        baseline_ids[0] = cls_id
        baseline_ids[-1] = sep_id
        baseline_ids[mask_positions] = mask_id

        baseline_t = torch.from_numpy(
            baseline_ids.astype(np.int64)
        ).unsqueeze(0).to(args.device)

        attr_result = lig.attribute(
            inputs=input_t,
            baselines=baseline_t,
            additional_forward_args=(attn_mask, first_mask_pos, actual_first_id),
            n_steps=args.n_steps,
            return_convergence_delta=True,
        )

        if isinstance(attr_result, tuple):
            attr_tensor, convergence_delta = attr_result
            conv_delta = convergence_delta.item()
        else:
            attr_tensor = attr_result
            conv_delta = float("nan")

        position_attrs = attr_tensor.sum(dim=-1).squeeze(0).cpu().numpy()

        # Surprisal at baseline
        with torch.no_grad():
            baseline_out = model(
                input_ids=baseline_t, attention_mask=attn_mask
            )
            baseline_lp = torch.log_softmax(
                baseline_out.logits[0, first_mask_pos], dim=-1
            )
            surprisal_baseline = -baseline_lp[actual_first_id].item()

        # Map subword attributions back to spaCy word tokens
        target_token_obj = doc[spacy_idx]
        pos_tag = target_token_obj.pos_

        context_records = []
        for ctx_spacy_idx, (ctx_sub_start, ctx_sub_end) in token_map.items():
            if ctx_spacy_idx == spacy_idx:
                continue

            ctx_start_w = ctx_sub_start - win_start
            ctx_end_w = ctx_sub_end - win_start
            ctx_start_w = max(0, ctx_start_w)
            ctx_end_w = min(window_len, ctx_end_w)

            if ctx_end_w <= ctx_start_w:
                continue

            if ctx_start_w < tok_end_w and ctx_end_w > tok_start_w:
                continue

            word_attr = position_attrs[ctx_start_w + 1: ctx_end_w + 1].sum()

            ctx_token = doc[ctx_spacy_idx]
            distance = ctx_spacy_idx - spacy_idx
            abs_distance = abs(distance)
            direction = "left" if distance < 0 else "right"

            context_records.append({
                "target_idx": i,
                "target_essay_id": essay_id,
                "target_text": row["token_text"],
                "target_surprisal": row["surprisal"],
                "target_pos": pos_tag,
                "prof_tertile": row["prof_tertile"],
                "ctx_spacy_idx": ctx_spacy_idx,
                "ctx_text": ctx_token.text,
                "ctx_pos": ctx_token.pos_,
                "ctx_pos_coarse": get_pos_coarse(ctx_token.pos_),
                "ctx_dep_rel": ctx_token.dep_,
                "distance": distance,
                "abs_distance": abs_distance,
                "direction": direction,
                "distance_band": get_distance_band(abs_distance),
                "attribution": float(word_attr),
            })

        attr_rows.extend(context_records)

        left_ctx = " ".join(t.text for t in doc[max(0, spacy_idx - 5):spacy_idx])
        right_ctx = " ".join(t.text for t in doc[spacy_idx + 1:spacy_idx + 6])

        meta_rows.append({
            "target_idx": i,
            "essay_id": essay_id,
            "spacy_idx": spacy_idx,
            "token_text": row["token_text"],
            "surprisal": row["surprisal"],
            "surprisal_baseline": surprisal_baseline,
            "pos_tag": pos_tag,
            "dep_rel": row["dep_rel"],
            "prof_tertile": row["prof_tertile"],
            "convergence_delta": conv_delta,
            "n_context_tokens": len(context_records),
            "left_context": left_ctx,
            "right_context": right_ctx,
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (n_total - i - 1)
            print(f"  [{i+1}/{n_total}] {elapsed:.0f}s elapsed, "
                  f"{rate:.2f}s/tok, ~{remaining:.0f}s remaining | "
                  f"conv_delta={conv_delta:.4f}")

    elapsed_total = time.time() - t0
    print(f"\nDone: {len(meta_rows)} tokens in {elapsed_total:.0f}s "
          f"({elapsed_total / max(1, len(meta_rows)):.2f}s/tok)")
    if n_skipped:
        print(f"Skipped: {n_skipped} tokens (missing essay or alignment)")

    cd = np.array([r["convergence_delta"] for r in meta_rows])
    cd_finite = cd[np.isfinite(cd)]
    if len(cd_finite) > 0:
        print(f"Convergence delta: mean={cd_finite.mean():.4f}, "
              f"std={cd_finite.std():.4f}, "
              f"median={np.median(cd_finite):.4f}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attrs_df = pd.DataFrame(attr_rows)
    attrs_df.to_parquet(OUTPUT_DIR / "nominal_ig_attrs.parquet", index=False)
    print(f"Saved {len(attrs_df)} attr rows → {OUTPUT_DIR / 'nominal_ig_attrs.parquet'}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(OUTPUT_DIR / "nominal_ig_meta.parquet", index=False)
    print(f"Saved {len(meta_df)} meta rows → {OUTPUT_DIR / 'nominal_ig_meta.parquet'}")

    params = {
        "method": "IntegratedGradients",
        "model": model_name,
        "baseline_type": args.baseline,
        "target": "-log_prob (surprisal)",
        "purpose": "nominal_ig_attribution",
        "target_pos": ["NOUN", "PROPN"],
        "n_per_tertile": args.n_per_tertile,
        "n_steps": args.n_steps,
        "window_size": args.window_size,
        "seed": args.seed,
        "device": args.device,
        "n_tokens_computed": len(meta_rows),
        "n_tokens_skipped": n_skipped,
        "n_attr_rows": len(attrs_df),
        "elapsed_seconds": round(elapsed_total, 1),
        "seconds_per_token": round(elapsed_total / max(1, len(meta_rows)), 3),
        "mean_convergence_delta": float(cd_finite.mean()) if len(cd_finite) > 0 else None,
    }
    with open(OUTPUT_DIR / "nominal_ig_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params → {OUTPUT_DIR / 'nominal_ig_params.json'}")


# ── analyze ──────────────────────────────────────────────────────────────

def cmd_analyze(args):
    """Compare nominal IG attribution profiles by proficiency tertile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    attrs_path = OUTPUT_DIR / "nominal_ig_attrs.parquet"
    meta_path = OUTPUT_DIR / "nominal_ig_meta.parquet"
    if not attrs_path.exists() or not meta_path.exists():
        print(f"ERROR: output files not found in {OUTPUT_DIR}")
        print("Run `python nominal_ig.py compute` first.")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path)

    print(f"Loaded {len(meta_df)} target tokens, {len(attrs_df)} attr rows")
    print(f"Tertiles: {meta_df['prof_tertile'].value_counts().to_dict()}")

    # ── Per-target metrics ────────────────────────────────────────────
    attrs_df["abs_attr"] = attrs_df["attribution"].abs()

    target_metrics = []
    for target_idx, grp in attrs_df.groupby("target_idx"):
        total_abs = grp["abs_attr"].sum()
        if total_abs == 0:
            continue

        meta_row = meta_df[meta_df["target_idx"] == target_idx].iloc[0]

        weighted_dist = (grp["abs_distance"] * grp["abs_attr"]).sum() / total_abs

        noun_abs = grp[grp["ctx_pos"].isin(["NOUN", "PROPN"])]["abs_attr"].sum()
        verb_abs = grp[grp["ctx_pos"].isin(["VERB", "AUX"])]["abs_attr"].sum()
        func_abs = grp[grp["ctx_pos"].isin(["DET", "ADP", "PRON"])]["abs_attr"].sum()

        left_abs = grp[grp["direction"] == "left"]["abs_attr"].sum()

        top3_abs = grp.nlargest(3, "abs_attr")["abs_attr"].sum()

        band_shares = {}
        for band in BAND_ORDER:
            band_abs = grp[grp["distance_band"] == band]["abs_attr"].sum()
            band_shares[f"share_{band}"] = band_abs / total_abs

        target_metrics.append({
            "target_idx": target_idx,
            "prof_tertile": meta_row["prof_tertile"],
            "surprisal": meta_row["surprisal"],
            "dep_rel": meta_row["dep_rel"],
            "ctx_noun_share": noun_abs / total_abs,
            "ctx_verb_share": verb_abs / total_abs,
            "ctx_func_share": func_abs / total_abs,
            "left_share": left_abs / total_abs,
            "weighted_mean_dist": weighted_dist,
            "top3_concentration": top3_abs / total_abs,
            **band_shares,
        })

    metrics_df = pd.DataFrame(target_metrics)
    low = metrics_df[metrics_df["prof_tertile"] == "low"]
    mid = metrics_df[metrics_df["prof_tertile"] == "mid"]
    high = metrics_df[metrics_df["prof_tertile"] == "high"]

    # ── Statistical tests ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("NOMINAL IG ATTRIBUTION: Proficiency comparison (low vs high)")
    print("=" * 80)

    test_vars = [
        ("ctx_noun_share", "NOUN/PROPN context share", "two-sided"),
        ("ctx_verb_share", "VERB/AUX context share", "two-sided"),
        ("ctx_func_share", "DET/ADP/PRON context share", "two-sided"),
        ("left_share", "Left (preceding) context share", "two-sided"),
        ("weighted_mean_dist", "Weighted mean attr distance", "two-sided"),
        ("top3_concentration", "Top-3 concentration", "two-sided"),
        ("share_1", "Band 1 |attr| share", "two-sided"),
        ("share_2-3", "Band 2-3 |attr| share", "two-sided"),
        ("share_4-7", "Band 4-7 |attr| share", "two-sided"),
        ("share_8+", "Band 8+ |attr| share", "two-sided"),
    ]

    print(f"\n{'Metric':<35s} {'Low':>8s} {'Mid':>8s} {'High':>8s} "
          f"{'L-H':>8s} {'U-stat':>10s} {'p':>10s} {'r_rb':>8s} {'Sig':>5s}")
    print("-" * 115)

    results = {}
    for var, label, alternative in test_vars:
        l_vals = low[var].dropna()
        m_vals = mid[var].dropna()
        h_vals = high[var].dropna()

        l_mean = l_vals.mean()
        m_mean = m_vals.mean()
        h_mean = h_vals.mean()
        diff = l_mean - h_mean

        U, p = mannwhitneyu(l_vals, h_vals, alternative=alternative)
        n1, n2 = len(l_vals), len(h_vals)
        r_rb = 1 - 2 * U / (n1 * n2)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"{label:<35s} {l_mean:>8.4f} {m_mean:>8.4f} {h_mean:>8.4f} "
              f"{diff:>+8.4f} {U:>10.0f} {p:>10.4f} {r_rb:>+8.3f} {sig:>5s}")

        results[var] = {
            "U": U, "p": p, "r_rb": r_rb, "sig": sig,
            "low_mean": l_mean, "mid_mean": m_mean, "high_mean": h_mean,
        }

    # ── Surprisal-stratified analysis ─────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SURPRISAL-STRATIFIED ANALYSIS (controlling for surprisal)")
    print("=" * 80)

    s_low, s_high = metrics_df["surprisal"].quantile([1/3, 2/3]).values
    metrics_df["surp_bin"] = pd.cut(
        metrics_df["surprisal"],
        bins=[-np.inf, s_low, s_high, np.inf],
        labels=["low_surp", "mid_surp", "high_surp"],
    )

    key_vars = ["ctx_noun_share", "ctx_verb_share", "ctx_func_share",
                "left_share", "weighted_mean_dist", "top3_concentration"]

    for surp_bin in ["low_surp", "mid_surp", "high_surp"]:
        sbin_df = metrics_df[metrics_df["surp_bin"] == surp_bin]
        sl = sbin_df[sbin_df["prof_tertile"] == "low"]
        sh = sbin_df[sbin_df["prof_tertile"] == "high"]
        n_l, n_h = len(sl), len(sh)

        print(f"\n--- {surp_bin} (n_low={n_l}, n_high={n_h}) ---")
        if n_l < 5 or n_h < 5:
            print("  Too few samples, skipping")
            continue

        print(f"  {'Metric':<30s} {'Low':>8s} {'High':>8s} {'p':>10s} {'r_rb':>8s}")
        for var in key_vars:
            lv = sl[var].dropna()
            hv = sh[var].dropna()
            if len(lv) < 3 or len(hv) < 3:
                continue
            U, p = mannwhitneyu(lv, hv, alternative="two-sided")
            r_rb = 1 - 2 * U / (len(lv) * len(hv))
            sig = "*" if p < 0.05 else ""
            print(f"  {var:<30s} {lv.mean():>8.4f} {hv.mean():>8.4f} "
                  f"{p:>10.4f} {r_rb:>+8.3f} {sig}")

    # ── Dependency relation analysis ──────────────────────────────────
    print(f"\n{'=' * 80}")
    print("TOP DEPENDENCY RELATIONS BY ATTRIBUTION SHARE")
    print("=" * 80)

    dep_shares = (
        attrs_df.groupby("ctx_dep_rel")["abs_attr"]
        .sum()
        .sort_values(ascending=False)
    )
    dep_total = dep_shares.sum()
    dep_top = dep_shares.head(15)

    print(f"\n{'Dep Rel':<15s} {'|attr| share':>12s} {'cumul':>8s}")
    cumul = 0
    for dep, val in dep_top.items():
        share = val / dep_total
        cumul += share
        print(f"{dep:<15s} {share:>12.4f} {cumul:>8.3f}")

    # By tertile
    print(f"\n{'Dep Rel':<12s} {'Low':>8s} {'Mid':>8s} {'High':>8s}")
    top_deps = dep_top.index[:10].tolist()
    for dep in top_deps:
        shares = {}
        for tertile in ["low", "mid", "high"]:
            tdf = attrs_df[attrs_df["prof_tertile"] == tertile]
            dep_abs = tdf[tdf["ctx_dep_rel"] == dep]["abs_attr"].sum()
            t_total = tdf["abs_attr"].sum()
            shares[tertile] = dep_abs / t_total if t_total > 0 else 0
        print(f"{dep:<12s} {shares['low']:>8.4f} {shares['mid']:>8.4f} "
              f"{shares['high']:>8.4f}")

    # ── Figure ────────────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else FIG_DIR / "nominal_ig.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Nominal Token IG Attribution Profiles by Proficiency", fontsize=13)

    tertile_colors = {"low": "#e74c3c", "mid": "#f39c12", "high": "#2ecc71"}
    tertile_labels = ["low", "mid", "high"]

    # Panel 1: Context POS share by proficiency
    ax = axes[0, 0]
    pos_vars = [
        ("ctx_noun_share", "NOUN"),
        ("ctx_verb_share", "VERB"),
        ("ctx_func_share", "FUNC"),
    ]
    x = np.arange(len(pos_vars))
    width = 0.25
    for j, tertile in enumerate(tertile_labels):
        tdf = metrics_df[metrics_df["prof_tertile"] == tertile]
        means = [tdf[v].mean() for v, _ in pos_vars]
        sems = [tdf[v].std() / np.sqrt(len(tdf)) for v, _ in pos_vars]
        ax.bar(x + j * width, means, width, yerr=sems, capsize=3,
               label=tertile, color=tertile_colors[tertile], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([lab for _, lab in pos_vars])
    ax.set_ylabel("|attr| share")
    ax.set_title("Context POS share")
    ax.legend(fontsize=8)

    # Panel 2: Left vs right by proficiency
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.25
    for j, tertile in enumerate(tertile_labels):
        tdf = metrics_df[metrics_df["prof_tertile"] == tertile]
        left_m = tdf["left_share"].mean()
        right_m = 1 - left_m
        left_se = tdf["left_share"].std() / np.sqrt(len(tdf))
        ax.bar(x + j * width, [left_m, right_m], width,
               yerr=[left_se, left_se], capsize=3,
               label=tertile, color=tertile_colors[tertile], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Left", "Right"])
    ax.set_ylabel("|attr| share")
    ax.set_title("Direction share")
    ax.legend(fontsize=8)

    # Panel 3: Top-3 concentration by proficiency
    ax = axes[0, 2]
    for tertile in tertile_labels:
        tdf = metrics_df[metrics_df["prof_tertile"] == tertile]
        ax.hist(tdf["top3_concentration"], bins=20, alpha=0.5,
                label=tertile, color=tertile_colors[tertile])
        ax.axvline(tdf["top3_concentration"].mean(),
                   color=tertile_colors[tertile], ls="--", lw=1.5)
    r = results.get("top3_concentration", {})
    p_val = r.get("p", 1.0)
    ax.set_title(f"Top-3 concentration (p={p_val:.4f})")
    ax.set_xlabel("|attr| share in top 3")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Panels 4-5: POS x distance heatmaps (low, high)
    for col_idx, (tertile, cmap) in enumerate([("low", "Reds"), ("high", "Blues")]):
        ax = axes[1, col_idx]
        gdf = attrs_df[attrs_df["prof_tertile"] == tertile].copy()
        pivot = gdf.pivot_table(
            values="abs_attr", index="ctx_pos_coarse",
            columns="distance_band", aggfunc="mean",
        ).reindex(index=POS_ORDER, columns=BAND_ORDER)

        data = pivot.values.astype(float)
        vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"{tertile.upper()} proficiency", fontsize=10)
        ax.set_xticks(range(len(BAND_ORDER)))
        ax.set_xticklabels(BAND_ORDER, fontsize=9)
        ax.set_xlabel("Distance band")
        ax.set_yticks(range(len(POS_ORDER)))
        ax.set_yticklabels(POS_ORDER, fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Context POS")

        for yi in range(data.shape[0]):
            for xi in range(data.shape[1]):
                val = data[yi, xi]
                if np.isnan(val):
                    ax.text(xi, yi, "—", ha="center", va="center",
                            fontsize=7, color="gray")
                else:
                    c = "white" if val > vmax * 0.6 else "black"
                    ax.text(xi, yi, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=c)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 6: Top dep_rels by attribution share
    ax = axes[1, 2]
    top_n_deps = 8
    dep_order = dep_shares.head(top_n_deps).index.tolist()
    x = np.arange(len(dep_order))
    width = 0.25
    for j, tertile in enumerate(tertile_labels):
        tdf = attrs_df[attrs_df["prof_tertile"] == tertile]
        t_total = tdf["abs_attr"].sum()
        shares = []
        for dep in dep_order:
            dep_abs = tdf[tdf["ctx_dep_rel"] == dep]["abs_attr"].sum()
            shares.append(dep_abs / t_total if t_total > 0 else 0)
        ax.bar(x + j * width, shares, width,
               label=tertile, color=tertile_colors[tertile], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(dep_order, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("|attr| share")
    ax.set_title("Top dep_rels by attribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"\nSaved figure → {output_path}")


# ── show ─────────────────────────────────────────────────────────────────

def cmd_show(args):
    """Inspect individual token attribution maps."""
    attrs_path = OUTPUT_DIR / "nominal_ig_attrs.parquet"
    meta_path = OUTPUT_DIR / "nominal_ig_meta.parquet"
    if not attrs_path.exists() or not meta_path.exists():
        print(f"ERROR: output files not found in {OUTPUT_DIR}")
        print("Run `python nominal_ig.py compute` first.")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path)

    attrs_df["abs_attr"] = attrs_df["attribution"].abs()

    # Select which tokens to show
    if args.target_idx:
        indices = args.target_idx
    else:
        # Auto-select: most local + most distributed from each tertile
        wmd = []
        for tidx, grp in attrs_df.groupby("target_idx"):
            total = grp["abs_attr"].sum()
            if total == 0:
                continue
            wd = (grp["abs_distance"] * grp["abs_attr"]).sum() / total
            top3 = grp.nlargest(3, "abs_attr")["abs_attr"].sum() / total
            wmd.append({"target_idx": tidx, "wmd": wd, "top3_conc": top3})
        wmd_df = pd.DataFrame(wmd).merge(
            meta_df[["target_idx", "prof_tertile"]], on="target_idx"
        )

        picks = []
        for tertile in ["low", "mid", "high"]:
            g = wmd_df[wmd_df["prof_tertile"] == tertile].sort_values("wmd")
            if len(g) >= 2:
                picks.append(int(g.iloc[0]["target_idx"]))   # most local
                picks.append(int(g.iloc[-1]["target_idx"]))  # most distributed
        indices = picks
        print(f"Auto-selected {len(indices)} tokens "
              f"(most local + most distributed per tertile)\n")

    for target_idx in indices:
        if target_idx not in meta_df["target_idx"].values:
            print(f"WARNING: target_idx {target_idx} not found, skipping\n")
            continue

        tmeta = meta_df[meta_df["target_idx"] == target_idx].iloc[0]
        tattrs = attrs_df[attrs_df["target_idx"] == target_idx].copy()
        tattrs = tattrs.sort_values("abs_attr", ascending=False)
        total_abs = tattrs["abs_attr"].sum()

        wmd_val = (
            (tattrs["abs_distance"] * tattrs["abs_attr"]).sum() / total_abs
            if total_abs > 0 else 0
        )

        print(f"{'=' * 70}")
        print(f"TARGET #{target_idx}  [{tmeta['prof_tertile'].upper()}]  "
              f"dep={tmeta['dep_rel']}")
        print(f"  Token: {tmeta['token_text']!r:16s}  POS: {tmeta['pos_tag']:<6s}  "
              f"surprisal: {tmeta['surprisal']:.3f}")
        print(f"  Convergence delta: {tmeta['convergence_delta']:.4f}  "
              f"Surprisal (baseline): {tmeta['surprisal_baseline']:.4f}  "
              f"Weighted mean dist: {wmd_val:.2f}")

        left = tmeta["left_context"] if isinstance(tmeta["left_context"], str) else ""
        right = tmeta["right_context"] if isinstance(tmeta["right_context"], str) else ""
        print(f"  ...{left}  [ {tmeta['token_text']} ]  {right}...")

        n_show = min(args.top_n, len(tattrs))
        print(f"\n  {'#':>3s}  {'DIST':>5s}  {'DIR':>5s}  {'TOKEN':<14s} "
              f"{'POS':<6s} {'DEP':<10s} {'ATTRIB':>8s}  {'|A|%':>6s}  {'CUMUL%':>6s}")
        cumul = 0.0
        for rank, (_, r) in enumerate(tattrs.head(n_show).iterrows(), 1):
            cumul += r["abs_attr"]
            pct = 100 * r["abs_attr"] / total_abs if total_abs > 0 else 0
            cpct = 100 * cumul / total_abs if total_abs > 0 else 0
            print(f"  {rank:>3d}  {r['distance']:>+5d}  {r['direction']:>5s}  "
                  f"{r['ctx_text']:<14s} {r['ctx_pos']:<6s} "
                  f"{r['ctx_dep_rel']:<10s} {r['attribution']:>+8.4f}  "
                  f"{pct:>5.1f}%  {cpct:>5.1f}%")

        # Band + POS summary
        band_abs = {b: tattrs[tattrs["distance_band"] == b]["abs_attr"].sum()
                    for b in BAND_ORDER}
        band_str = "  Bands: " + "  ".join(
            f"{b}={100*v/total_abs:.0f}%" if total_abs > 0 else f"{b}=0%"
            for b, v in band_abs.items()
        )
        print(band_str)

        noun_share = tattrs[tattrs["ctx_pos"].isin(["NOUN", "PROPN"])]["abs_attr"].sum()
        verb_share = tattrs[tattrs["ctx_pos"].isin(["VERB", "AUX"])]["abs_attr"].sum()
        left_share = tattrs[tattrs["direction"] == "left"]["abs_attr"].sum()
        if total_abs > 0:
            print(f"  NOUN ctx: {100*noun_share/total_abs:.1f}%  "
                  f"VERB ctx: {100*verb_share/total_abs:.1f}%  "
                  f"Left: {100*left_share/total_abs:.1f}%")
        print()


# ── parser ───────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Integrated Gradients attribution profiling for nominal tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compute
    p_compute = sub.add_parser(
        "compute", help="Sample nominal tokens, run IG"
    )
    p_compute.add_argument(
        "--model", choices=["bert", "modernbert"], default="modernbert",
        help="Model to use (default: modernbert)"
    )
    p_compute.add_argument(
        "--baseline", choices=["pad", "mask"], default="mask",
        help="Baseline type: pad ([PAD] token) or mask ([MASK] token) "
             "(default: mask)"
    )
    p_compute.add_argument(
        "--n-steps", type=int, default=100,
        help="IG interpolation steps (default: 100)"
    )
    p_compute.add_argument(
        "--n-per-tertile", type=int, default=200,
        help="Tokens to sample per proficiency tertile (default: 200)"
    )
    p_compute.add_argument(
        "--window-size", type=int, default=64,
        help="Context window in subword tokens (default: 64)"
    )
    p_compute.add_argument("--seed", type=int, default=42)
    p_compute.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)"
    )

    # analyze
    p_analyze = sub.add_parser(
        "analyze", help="Compare attribution profiles by proficiency"
    )
    p_analyze.add_argument(
        "--output", type=str, default=None,
        help="Output PDF path (default: fig/nominal_ig.pdf)"
    )

    # show
    p_show = sub.add_parser(
        "show", help="Inspect individual token attribution maps"
    )
    p_show.add_argument(
        "target_idx", type=int, nargs="*", default=None,
        help="Target indices to inspect (default: auto-select extremes)"
    )
    p_show.add_argument(
        "--top-n", type=int, default=12,
        help="Number of top attributions to show per token (default: 12)"
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compute":
        cmd_compute(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "show":
        cmd_show(args)
