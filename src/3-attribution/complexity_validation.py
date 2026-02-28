"""AttnLRP method validation via syntactic complexity.

Validates that AttnLRP can detect *known* structural variation: essays with
longer syntactic dependencies (high mean dependency distance) should produce
attribution profiles shifted toward longer distances.

If this test fails, the null proficiency result from lrp_pilot may reflect a
method limitation (locality bias) rather than a genuine finding.

Subcommands:
    compute  — Select extreme-complexity essays, sample tokens, run AttnLRP
    analyze  — Compare attribution profiles by complexity group

Example workflow:
    python complexity_validation.py compute --device cuda
    python complexity_validation.py analyze

Source data:
    data/ellipse_docbins/processed_docs_00.spacy — pre-parsed spaCy docs
    data/ELLIPSE_Final_github.csv — essay texts + scores
    data/ELLIPSE_token_predictability.parquet — token-level surprisal

Outputs (in data/lrp_pilot/):
    essay_complexity.parquet — per-essay mean dependency distance (cached)
    complexity_attrs.parquet — one row per (target, context) pair
    complexity_meta.parquet  — one row per target token
    complexity_params.json   — hyperparameters and runtime stats
    fig/complexity_validation.pdf — comparison figure
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

# Reuse helpers from lrp_pilot
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
COMPLEXITY_CACHE = OUTPUT_DIR / "essay_complexity.parquet"

MODELS = {
    "bert": "bert-base-uncased",
    "modernbert": "answerdotai/ModernBERT-base",
}


# ── compute ──────────────────────────────────────────────────────────────

def cmd_compute(args):
    """Select extreme-complexity essays, sample tokens, run AttnLRP."""
    import spacy
    import torch
    from spacy.tokens import DocBin
    from transformers import AutoTokenizer, AutoConfig
    from features.predictability import get_centered_window
    from features.syntactic import mean_dep_distance

    # ── Step 1: Compute essay-level syntactic complexity ──────────────
    if COMPLEXITY_CACHE.exists() and not args.recompute:
        print(f"Loading cached complexity from {COMPLEXITY_CACHE}")
        complexity_df = pd.read_parquet(COMPLEXITY_CACHE)
    else:
        print(f"Computing essay-level complexity from {DOCBIN_PATH}...")
        nlp = spacy.load("en_core_web_lg")
        doc_bin = DocBin().from_disk(DOCBIN_PATH)
        docs = list(doc_bin.get_docs(nlp.vocab))
        print(f"  Loaded {len(docs)} docs from DocBin")

        # Load essay texts for matching
        essays_df = pd.read_csv(ELLIPSE_PATH)

        # Build alnum-normalized lookup: norm_text[:80] -> text_id_kaggle
        def alnum_key(text):
            return re.sub(r"[^a-zA-Z0-9]", "", text)[:80]

        essay_lookup = {}
        for _, row in essays_df.iterrows():
            key = alnum_key(row["full_text"])
            essay_lookup[key] = row["text_id_kaggle"]

        # Match docs to essay IDs
        complexity_rows = []
        n_matched = 0
        for doc in docs:
            key = alnum_key(doc.text)
            if key in essay_lookup:
                eid = essay_lookup[key]
                mdd = mean_dep_distance(doc)
                n_tok = len(doc)
                complexity_rows.append({
                    "text_id_kaggle": eid,
                    "mean_dep_distance": mdd,
                    "n_tokens": n_tok,
                })
                n_matched += 1

        complexity_df = pd.DataFrame(complexity_rows)
        print(f"  Matched {n_matched}/{len(essays_df)} essays")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        complexity_df.to_parquet(COMPLEXITY_CACHE, index=False)
        print(f"  Cached → {COMPLEXITY_CACHE}")

    # ── Step 2: Select extreme deciles ────────────────────────────────
    q_low = complexity_df["mean_dep_distance"].quantile(0.10)
    q_high = complexity_df["mean_dep_distance"].quantile(0.90)

    simple_essays = set(
        complexity_df[complexity_df["mean_dep_distance"] <= q_low]["text_id_kaggle"]
    )
    complex_essays = set(
        complexity_df[complexity_df["mean_dep_distance"] >= q_high]["text_id_kaggle"]
    )
    print(f"\nDecile thresholds: simple ≤ {q_low:.3f}, complex ≥ {q_high:.3f}")
    print(f"  Simple essays: {len(simple_essays)}, Complex essays: {len(complex_essays)}")

    # ── Step 3: Load tokens, filter, sample ───────────────────────────
    print(f"\nLoading token data from {FULL_TOKEN_PATH}...")
    full_tokens = pd.read_parquet(FULL_TOKEN_PATH)
    full_tokens = full_tokens.rename(columns={
        "text_id_kaggle": "essay_id",
        "text": "token_text",
        "mean_loss": "surprisal",
    })

    # Merge complexity info
    complexity_lookup = complexity_df.set_index("text_id_kaggle")["mean_dep_distance"].to_dict()

    # Assign complexity group
    full_tokens["complexity_group"] = None
    full_tokens.loc[
        full_tokens["essay_id"].isin(simple_essays), "complexity_group"
    ] = "simple"
    full_tokens.loc[
        full_tokens["essay_id"].isin(complex_essays), "complexity_group"
    ] = "complex"

    # Filter to selected essays only
    sel = full_tokens[full_tokens["complexity_group"].notna()].copy()
    print(f"  Tokens in extreme deciles: {len(sel)}")

    # Filter to well-predicted tokens (low surprisal)
    sel = sel[sel["surprisal"] <= args.max_surprisal]
    print(f"  After surprisal ≤ {args.max_surprisal} filter: {len(sel)}")

    # Exclude first/last token of each essay
    essay_lengths = sel.groupby("essay_id")["spacy_idx"].transform("max")
    sel = sel[(sel["spacy_idx"] > 0) & (sel["spacy_idx"] < essay_lengths)]
    print(f"  After excluding boundary tokens: {len(sel)}")

    # Sample N per group
    rng = np.random.default_rng(args.seed)
    samples = []
    for group in ["simple", "complex"]:
        grp_df = sel[sel["complexity_group"] == group]
        n = min(args.n_per_group, len(grp_df))
        idx = rng.choice(len(grp_df), size=n, replace=False)
        samples.append(grp_df.iloc[idx])
    sample_df = pd.concat(samples).reset_index(drop=True)

    # Add essay-level complexity
    sample_df["essay_mean_dep_distance"] = sample_df["essay_id"].map(complexity_lookup)

    print(f"\nSampled {len(sample_df)} tokens: "
          f"{sample_df['complexity_group'].value_counts().to_dict()}")

    # ── Step 4: Run AttnLRP ───────────────────────────────────────────
    model_key = args.model
    model_name = MODELS[model_key]

    if model_key == "bert":
        import transformers.models.bert.modeling_bert as modeling_mod
        from lxt.efficient import monkey_patch
        monkey_patch(modeling_mod, verbose=True)
        ModelClass = modeling_mod.BertForMaskedLM
    else:
        import transformers.models.modernbert.modeling_modernbert as modeling_mod
        from modernbert_lrp import monkey_patch_modernbert
        monkey_patch_modernbert(modeling_mod, verbose=True)
        ModelClass = modeling_mod.ModernBertForMaskedLM

    print(f"\nLoading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    model = ModelClass.from_pretrained(model_name, config=config)
    model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load spaCy for reparsing
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_lg")

    # Load essays
    print("Loading essays...")
    essays_df = pd.read_csv(ELLIPSE_PATH)
    essay_texts = dict(zip(essays_df["text_id_kaggle"], essays_df["full_text"]))

    # Special tokens
    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if model_key == "bert":
        embed_fn = model.bert.get_input_embeddings()
    else:
        embed_fn = model.model.embeddings.tok_embeddings

    # Essay cache
    essay_cache = {}

    def get_essay_data(essay_id):
        if essay_id not in essay_cache:
            text = essay_texts[essay_id]
            doc = nlp(text)
            token_map, trf_tok_ids = get_token_alignment(tokenizer, doc)
            essay_cache[essay_id] = (doc, token_map, trf_tok_ids)
        return essay_cache[essay_id]

    # Attribution loop
    attr_rows = []
    meta_rows = []
    n_skipped = 0
    conservation_ratios = []

    t0 = time.time()
    n_total = len(sample_df)
    print(f"\nComputing AttnLRP for {n_total} tokens...")

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

        window_ids[tok_start_w:tok_end_w] = mask_id

        full_ids = np.concatenate([[cls_id], window_ids, [sep_id]])
        first_mask_pos = tok_start_w + 1
        actual_first_id = int(trf_tok_ids[subword_start])

        input_ids = torch.from_numpy(
            full_ids.astype(np.int64)
        ).unsqueeze(0).to(args.device)
        attn_mask = torch.ones_like(input_ids)

        input_embeds = embed_fn(input_ids).detach().requires_grad_(True)

        outputs = model(inputs_embeds=input_embeds, attention_mask=attn_mask)
        logits = outputs.logits[0, first_mask_pos]

        # Backward from raw logit (preserves LRP conservation)
        target_scalar = logits[actual_first_id]
        target_scalar.backward()

        relevance = (input_embeds * input_embeds.grad).sum(-1).squeeze(0)

        target_val = target_scalar.item()
        rel_sum = relevance.sum().item()
        conservation = rel_sum / target_val if abs(target_val) > 1e-8 else float("nan")
        conservation_ratios.append(conservation)

        position_rels = relevance.detach().cpu().numpy()
        input_embeds.grad = None

        target_token_obj = doc[spacy_idx]
        pos_tag = target_token_obj.pos_

        # Map subword relevances back to spaCy tokens
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

            word_rel = position_rels[ctx_start_w + 1: ctx_end_w + 1].sum()

            ctx_token = doc[ctx_spacy_idx]
            distance = ctx_spacy_idx - spacy_idx
            abs_distance = abs(distance)

            context_records.append({
                "target_idx": i,
                "target_essay_id": essay_id,
                "target_text": row["token_text"],
                "target_surprisal": row["surprisal"],
                "target_pos": pos_tag,
                "complexity_group": row["complexity_group"],
                "essay_mean_dep_distance": row["essay_mean_dep_distance"],
                "ctx_spacy_idx": ctx_spacy_idx,
                "ctx_text": ctx_token.text,
                "ctx_pos": ctx_token.pos_,
                "ctx_pos_coarse": get_pos_coarse(ctx_token.pos_),
                "distance": distance,
                "abs_distance": abs_distance,
                "distance_band": get_distance_band(abs_distance),
                "attribution": float(word_rel),
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
            "target_scalar": target_val,
            "pos_tag": pos_tag,
            "complexity_group": row["complexity_group"],
            "essay_mean_dep_distance": row["essay_mean_dep_distance"],
            "conservation_ratio": conservation,
            "n_context_tokens": len(context_records),
            "left_context": left_ctx,
            "right_context": right_ctx,
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (n_total - i - 1)
            print(f"  [{i+1}/{n_total}] {elapsed:.0f}s elapsed, "
                  f"{rate:.3f}s/tok, ~{remaining:.0f}s remaining | "
                  f"conservation={conservation:.4f}")

    elapsed_total = time.time() - t0
    print(f"\nDone: {len(meta_rows)} tokens in {elapsed_total:.0f}s "
          f"({elapsed_total / max(1, len(meta_rows)):.3f}s/tok)")
    if n_skipped:
        print(f"Skipped: {n_skipped} tokens (missing essay or alignment)")

    cr = np.array(conservation_ratios)
    cr_finite = cr[np.isfinite(cr)]
    if len(cr_finite) > 0:
        print(f"Conservation ratio: mean={cr_finite.mean():.4f}, "
              f"std={cr_finite.std():.4f}, "
              f"median={np.median(cr_finite):.4f}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attrs_df = pd.DataFrame(attr_rows)
    attrs_df.to_parquet(OUTPUT_DIR / "complexity_attrs.parquet", index=False)
    print(f"Saved {len(attrs_df)} attr rows → {OUTPUT_DIR / 'complexity_attrs.parquet'}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(OUTPUT_DIR / "complexity_meta.parquet", index=False)
    print(f"Saved {len(meta_df)} meta rows → {OUTPUT_DIR / 'complexity_meta.parquet'}")

    params = {
        "method": "AttnLRP",
        "model": model_name,
        "purpose": "complexity_validation",
        "n_per_group": args.n_per_group,
        "max_surprisal": args.max_surprisal,
        "window_size": args.window_size,
        "seed": args.seed,
        "device": args.device,
        "decile_low": float(q_low),
        "decile_high": float(q_high),
        "n_simple_essays": len(simple_essays),
        "n_complex_essays": len(complex_essays),
        "n_tokens_computed": len(meta_rows),
        "n_tokens_skipped": n_skipped,
        "n_attr_rows": len(attrs_df),
        "elapsed_seconds": round(elapsed_total, 1),
        "seconds_per_token": round(elapsed_total / max(1, len(meta_rows)), 3),
        "mean_conservation_ratio": float(cr_finite.mean()) if len(cr_finite) > 0 else None,
    }
    with open(OUTPUT_DIR / "complexity_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params → {OUTPUT_DIR / 'complexity_params.json'}")


# ── analyze ──────────────────────────────────────────────────────────────

def cmd_analyze(args):
    """Compare attribution profiles by syntactic complexity group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.stats import mannwhitneyu

    attrs_path = OUTPUT_DIR / "complexity_attrs.parquet"
    meta_path = OUTPUT_DIR / "complexity_meta.parquet"
    if not attrs_path.exists() or not meta_path.exists():
        print(f"ERROR: output files not found in {OUTPUT_DIR}")
        print("Run `python complexity_validation.py compute` first.")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path)

    print(f"Loaded {len(meta_df)} target tokens, {len(attrs_df)} attr rows")
    print(f"Groups: {meta_df['complexity_group'].value_counts().to_dict()}")
    print(f"Mean dep distance — simple: "
          f"{meta_df[meta_df['complexity_group']=='simple']['essay_mean_dep_distance'].mean():.3f}, "
          f"complex: "
          f"{meta_df[meta_df['complexity_group']=='complex']['essay_mean_dep_distance'].mean():.3f}")

    # ── Per-target metrics ────────────────────────────────────────────
    attrs_df["abs_attr"] = attrs_df["attribution"].abs()

    target_metrics = []
    for target_idx, grp in attrs_df.groupby("target_idx"):
        total_abs = grp["abs_attr"].sum()
        if total_abs == 0:
            continue

        meta_row = meta_df[meta_df["target_idx"] == target_idx].iloc[0]

        # Weighted mean attribution distance
        weighted_dist = (grp["abs_distance"] * grp["abs_attr"]).sum() / total_abs

        # Band shares
        band_shares = {}
        for band in BAND_ORDER:
            band_abs = grp[grp["distance_band"] == band]["abs_attr"].sum()
            band_shares[f"share_{band}"] = band_abs / total_abs

        # Content word (NOUN, VERB) attribution at d >= 4
        content_far = grp[
            (grp["ctx_pos"].isin(["NOUN", "PROPN", "VERB", "AUX"]))
            & (grp["abs_distance"] >= 4)
        ]
        content_far_share = content_far["abs_attr"].sum() / total_abs

        target_metrics.append({
            "target_idx": target_idx,
            "complexity_group": meta_row["complexity_group"],
            "essay_mean_dep_distance": meta_row["essay_mean_dep_distance"],
            "surprisal": meta_row["surprisal"],
            "weighted_mean_distance": weighted_dist,
            "content_far_share": content_far_share,
            **band_shares,
        })

    metrics_df = pd.DataFrame(target_metrics)
    simple = metrics_df[metrics_df["complexity_group"] == "simple"]
    cplx = metrics_df[metrics_df["complexity_group"] == "complex"]

    # ── Statistical tests ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLEXITY VALIDATION: AttnLRP attribution profiles")
    print("=" * 70)

    test_vars = [
        ("weighted_mean_distance", "Weighted mean attr distance", "greater"),
        ("share_4-7", "Band 4-7 |attr| share", "greater"),
        ("share_8+", "Band 8+ |attr| share", "greater"),
        ("content_far_share", "Content words (NOUN/VERB) at d≥4", "greater"),
        ("share_1", "Band 1 |attr| share", "less"),
    ]

    print(f"\n{'Metric':<40s} {'Simple':>8s} {'Complex':>8s} "
          f"{'Diff':>8s} {'U-stat':>10s} {'p':>10s} {'r_rb':>8s} {'Sig':>5s}")
    print("-" * 100)

    results = {}
    for var, label, alternative in test_vars:
        s_vals = simple[var].dropna()
        c_vals = cplx[var].dropna()

        s_mean = s_vals.mean()
        c_mean = c_vals.mean()
        diff = c_mean - s_mean

        U, p = mannwhitneyu(c_vals, s_vals, alternative=alternative)
        # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
        n1, n2 = len(c_vals), len(s_vals)
        r_rb = 1 - 2 * U / (n1 * n2)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"{label:<40s} {s_mean:>8.4f} {c_mean:>8.4f} "
              f"{diff:>+8.4f} {U:>10.0f} {p:>10.4f} {r_rb:>+8.3f} {sig:>5s}")

        results[var] = {"U": U, "p": p, "r_rb": r_rb, "simple_mean": s_mean,
                        "complex_mean": c_mean, "sig": sig}

    # ── Diagnostic: surprisal balance ─────────────────────────────────
    print(f"\n--- Surprisal balance check ---")
    print(f"  Simple mean surprisal:  {simple['surprisal'].mean():.3f} "
          f"(sd={simple['surprisal'].std():.3f})")
    print(f"  Complex mean surprisal: {cplx['surprisal'].mean():.3f} "
          f"(sd={cplx['surprisal'].std():.3f})")
    U_surp, p_surp = mannwhitneyu(simple["surprisal"], cplx["surprisal"],
                                   alternative="two-sided")
    print(f"  Mann-Whitney p={p_surp:.4f} (two-sided; want p>0.05 = balanced)")

    # ── Interpretation ────────────────────────────────────────────────
    primary_pass = results["weighted_mean_distance"]["p"] < 0.05
    secondary_pass = results["share_4-7"]["p"] < 0.05

    print(f"\n{'=' * 70}")
    if primary_pass:
        print("VALIDATION PASSED: AttnLRP detects systematic complexity differences.")
        print(f"  Weighted distance: complex {results['weighted_mean_distance']['complex_mean']:.3f} "
              f"> simple {results['weighted_mean_distance']['simple_mean']:.3f} "
              f"(p={results['weighted_mean_distance']['p']:.4f})")
    else:
        print("VALIDATION FAILED: AttnLRP does NOT detect complexity differences.")
        print("  The null proficiency result may reflect a method limitation.")

    if secondary_pass:
        print(f"  Band 4-7 share also higher for complex essays "
              f"(p={results['share_4-7']['p']:.4f})")
    print("=" * 70)

    # ── Figure ────────────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else FIG_DIR / "complexity_validation.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("AttnLRP Complexity Validation: Simple vs Complex Essays", fontsize=13)

    # Panel 1: Weighted mean distance distributions
    ax = axes[0, 0]
    bins = np.linspace(
        min(simple["weighted_mean_distance"].min(), cplx["weighted_mean_distance"].min()),
        max(simple["weighted_mean_distance"].max(), cplx["weighted_mean_distance"].max()),
        25,
    )
    ax.hist(simple["weighted_mean_distance"], bins=bins, alpha=0.6, label="Simple", color="steelblue")
    ax.hist(cplx["weighted_mean_distance"], bins=bins, alpha=0.6, label="Complex", color="coral")
    ax.axvline(simple["weighted_mean_distance"].mean(), color="steelblue", ls="--", lw=1.5)
    ax.axvline(cplx["weighted_mean_distance"].mean(), color="coral", ls="--", lw=1.5)
    r = results["weighted_mean_distance"]
    ax.set_title(f"Weighted mean attr distance\np={r['p']:.4f}, r={r['r_rb']:+.3f}")
    ax.set_xlabel("Weighted mean distance")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # Panel 2: Band shares comparison (grouped bar)
    ax = axes[0, 1]
    x = np.arange(len(BAND_ORDER))
    width = 0.35
    s_shares = [simple[f"share_{b}"].mean() for b in BAND_ORDER]
    c_shares = [cplx[f"share_{b}"].mean() for b in BAND_ORDER]
    ax.bar(x - width / 2, s_shares, width, label="Simple", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, c_shares, width, label="Complex", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_ORDER)
    ax.set_xlabel("Distance band")
    ax.set_ylabel("Mean |attr| share")
    ax.set_title("Band share comparison")
    ax.legend(fontsize=9)

    # Panel 3: Content-word far attribution
    ax = axes[0, 2]
    ax.hist(simple["content_far_share"], bins=20, alpha=0.6, label="Simple", color="steelblue")
    ax.hist(cplx["content_far_share"], bins=20, alpha=0.6, label="Complex", color="coral")
    ax.axvline(simple["content_far_share"].mean(), color="steelblue", ls="--", lw=1.5)
    ax.axvline(cplx["content_far_share"].mean(), color="coral", ls="--", lw=1.5)
    r = results["content_far_share"]
    ax.set_title(f"Content words at d≥4\np={r['p']:.4f}, r={r['r_rb']:+.3f}")
    ax.set_xlabel("|attr| share")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # Panel 4-5: POS × distance heatmaps per group
    for col_idx, (group, label, color) in enumerate([
        ("simple", "Simple (bottom decile)", "Blues"),
        ("complex", "Complex (top decile)", "Reds"),
    ]):
        ax = axes[1, col_idx]
        gdf = attrs_df[attrs_df["complexity_group"] == group]
        gdf = gdf.copy()
        gdf["abs_attr"] = gdf["attribution"].abs()
        pivot = gdf.pivot_table(
            values="abs_attr",
            index="ctx_pos_coarse",
            columns="distance_band",
            aggfunc="mean",
        ).reindex(index=POS_ORDER, columns=BAND_ORDER)

        data = pivot.values.astype(float)
        vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1
        im = ax.imshow(data, cmap=color, aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(BAND_ORDER)))
        ax.set_xticklabels(BAND_ORDER, fontsize=9)
        ax.set_xlabel("Distance band")
        if col_idx == 0:
            ax.set_yticks(range(len(POS_ORDER)))
            ax.set_yticklabels(POS_ORDER, fontsize=9)
            ax.set_ylabel("Context POS")
        else:
            ax.set_yticks(range(len(POS_ORDER)))
            ax.set_yticklabels(POS_ORDER, fontsize=9)

        for yi in range(data.shape[0]):
            for xi in range(data.shape[1]):
                val = data[yi, xi]
                if np.isnan(val):
                    ax.text(xi, yi, "—", ha="center", va="center", fontsize=7, color="gray")
                else:
                    c = "white" if val > vmax * 0.6 else "black"
                    ax.text(xi, yi, f"{val:.3f}", ha="center", va="center", fontsize=7, color=c)

        fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 6: Difference heatmap (complex - simple)
    ax = axes[1, 2]
    simple_pivot = attrs_df[attrs_df["complexity_group"] == "simple"].copy()
    simple_pivot["abs_attr"] = simple_pivot["attribution"].abs()
    simple_pivot = simple_pivot.pivot_table(
        values="abs_attr", index="ctx_pos_coarse", columns="distance_band", aggfunc="mean",
    ).reindex(index=POS_ORDER, columns=BAND_ORDER)

    complex_pivot = attrs_df[attrs_df["complexity_group"] == "complex"].copy()
    complex_pivot["abs_attr"] = complex_pivot["attribution"].abs()
    complex_pivot = complex_pivot.pivot_table(
        values="abs_attr", index="ctx_pos_coarse", columns="distance_band", aggfunc="mean",
    ).reindex(index=POS_ORDER, columns=BAND_ORDER)

    diff = complex_pivot - simple_pivot
    data = diff.values.astype(float)
    vmax = np.nanmax(np.abs(data)) if np.any(np.isfinite(data)) else 1
    if vmax == 0:
        vmax = 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_title("Complex − Simple (diff)", fontsize=10)
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels(BAND_ORDER, fontsize=9)
    ax.set_xlabel("Distance band")
    ax.set_yticks(range(len(POS_ORDER)))
    ax.set_yticklabels(POS_ORDER, fontsize=9)

    for yi in range(data.shape[0]):
        for xi in range(data.shape[1]):
            val = data[yi, xi]
            if np.isnan(val):
                ax.text(xi, yi, "—", ha="center", va="center", fontsize=7, color="gray")
            else:
                c = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(xi, yi, f"{val:+.3f}", ha="center", va="center", fontsize=7, color=c)

    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"\nSaved figure → {output_path}")


# ── show ─────────────────────────────────────────────────────────────────

def cmd_show(args):
    """Inspect individual token attribution maps."""
    attrs_path = OUTPUT_DIR / "complexity_attrs.parquet"
    meta_path = OUTPUT_DIR / "complexity_meta.parquet"
    if not attrs_path.exists() or not meta_path.exists():
        print(f"ERROR: output files not found in {OUTPUT_DIR}")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path)

    # Select which tokens to show
    if args.target_idx:
        indices = args.target_idx
    else:
        # Auto-select: pick tokens with highest and lowest weighted mean
        # attribution distance from each group for contrast
        attrs_df["abs_attr"] = attrs_df["attribution"].abs()
        wmd = []
        for tidx, grp in attrs_df.groupby("target_idx"):
            total = grp["abs_attr"].sum()
            if total == 0:
                continue
            wd = (grp["abs_distance"] * grp["abs_attr"]).sum() / total
            wmd.append({"target_idx": tidx, "wmd": wd})
        wmd_df = pd.DataFrame(wmd).merge(
            meta_df[["target_idx", "complexity_group"]], on="target_idx"
        )

        picks = []
        for group in ["simple", "complex"]:
            g = wmd_df[wmd_df["complexity_group"] == group].sort_values("wmd")
            # Most local and most distant from each group
            picks.append(int(g.iloc[0]["target_idx"]))
            picks.append(int(g.iloc[-1]["target_idx"]))
        indices = picks
        print(f"Auto-selected {len(indices)} tokens (most local + most distant per group)\n")

    for target_idx in indices:
        if target_idx not in meta_df["target_idx"].values:
            print(f"WARNING: target_idx {target_idx} not found, skipping\n")
            continue

        tmeta = meta_df[meta_df["target_idx"] == target_idx].iloc[0]
        tattrs = attrs_df[attrs_df["target_idx"] == target_idx].copy()
        tattrs["abs_attr"] = tattrs["attribution"].abs()
        tattrs = tattrs.sort_values("abs_attr", ascending=False)
        total_abs = tattrs["abs_attr"].sum()

        # Weighted mean distance for this token
        wmd_val = (tattrs["abs_distance"] * tattrs["abs_attr"]).sum() / total_abs if total_abs > 0 else 0

        print(f"{'=' * 70}")
        print(f"TARGET #{target_idx}  [{tmeta['complexity_group'].upper()}]  "
              f"essay_mdd={tmeta['essay_mean_dep_distance']:.2f}")
        print(f"  Token: {tmeta['token_text']!r:16s}  POS: {tmeta['pos_tag']:<6s}  "
              f"surprisal: {tmeta['surprisal']:.3f}")
        print(f"  Conservation: {tmeta['conservation_ratio']:.4f}  "
              f"Logit: {tmeta['target_scalar']:.4f}  "
              f"Weighted mean dist: {wmd_val:.2f}")

        # KWIC
        left = tmeta["left_context"] if isinstance(tmeta["left_context"], str) else ""
        right = tmeta["right_context"] if isinstance(tmeta["right_context"], str) else ""
        print(f"  ...{left}  [ {tmeta['token_text']} ]  {right}...")

        # Top attributions
        n_show = min(args.top_n, len(tattrs))
        print(f"\n  {'#':>3s}  {'DIST':>5s}  {'TOKEN':<14s} {'POS':<6s} "
              f"{'ATTRIB':>8s}  {'|A|%':>6s}  {'CUMUL%':>6s}")
        cumul = 0.0
        for rank, (_, r) in enumerate(tattrs.head(n_show).iterrows(), 1):
            cumul += r["abs_attr"]
            pct = 100 * r["abs_attr"] / total_abs if total_abs > 0 else 0
            cpct = 100 * cumul / total_abs if total_abs > 0 else 0
            print(f"  {rank:>3d}  {r['distance']:>+5d}  {r['ctx_text']:<14s} "
                  f"{r['ctx_pos']:<6s} {r['attribution']:>+8.4f}  "
                  f"{pct:>5.1f}%  {cpct:>5.1f}%")

        # Band summary
        band_abs = {b: tattrs[tattrs["distance_band"] == b]["abs_attr"].sum() for b in BAND_ORDER}
        band_str = "  Bands: " + "  ".join(
            f"{b}={100*v/total_abs:.0f}%" if total_abs > 0 else f"{b}=0%"
            for b, v in band_abs.items()
        )
        print(band_str)
        print()


# ── parser ───────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="AttnLRP method validation via syntactic complexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compute
    p_compute = sub.add_parser(
        "compute", help="Select extreme-complexity essays, sample tokens, run AttnLRP"
    )
    p_compute.add_argument(
        "--model", choices=["bert", "modernbert"], default="modernbert",
        help="Model to use (default: modernbert)"
    )
    p_compute.add_argument(
        "--n-per-group", type=int, default=150,
        help="Tokens to sample per complexity group (default: 150)"
    )
    p_compute.add_argument(
        "--window-size", type=int, default=64,
        help="Context window in subword tokens (default: 64)"
    )
    p_compute.add_argument(
        "--max-surprisal", type=float, default=1.0,
        help="Filter to well-predicted tokens with surprisal ≤ this (default: 1.0)"
    )
    p_compute.add_argument("--seed", type=int, default=42)
    p_compute.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)"
    )
    p_compute.add_argument(
        "--recompute", action="store_true",
        help="Recompute essay complexity even if cache exists"
    )

    # analyze
    p_analyze = sub.add_parser(
        "analyze", help="Compare attribution profiles by complexity group"
    )
    p_analyze.add_argument(
        "--output", type=str, default=None,
        help="Output PDF path (default: fig/complexity_validation.pdf)"
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
