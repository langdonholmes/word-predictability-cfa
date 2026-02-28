"""Leave-one-out (LOO) attribution pilot for decomposing token-level surprisal.

Computes per-context-token attribution scores for ALL tokens in sampled ELLIPSE
essays using double-masked LOO.  Both the baseline and the LOO comparison use
genuine MLM predictions (logits at [MASK] positions only).

Method:
    For each target token i:
    1. Baseline: mask only token i → MLM surprisal at i (true masked prediction).
    2. LOO: for each context token j≠i, mask BOTH i and j → MLM surprisal at i.
    3. Δ_ij = surprisal(i | i+j masked) − surprisal(i | only i masked).
       Positive Δ → j was helping predict i (removing j raised surprisal).
       Negative Δ → j was misleading (removing j lowered surprisal).

    Cost: L baseline passes + L(L−1)/2 pairwise passes (each fills both
    directions) = O(L²) forward passes per document, batched for GPU
    efficiency.

Subcommands:
    compute  — Sample documents and compute LOO attributions
    show     — Inspect a single token's attribution map
    profile  — Aggregate attribution profiles by proficiency

Example workflow:
    python loo_pilot.py compute --n-docs-per-tertile 5
    python loo_pilot.py show 0 5
    python loo_pilot.py show 0 20 --top-n 20
    python loo_pilot.py profile
    python loo_pilot.py profile --abs
    python loo_pilot.py profile --target-filter high-surprisal

Source data (read-only):
    data/ELLIPSE_Final_github.csv — essay texts + proficiency scores

Outputs (in data/loo_pilot/):
    loo_pilot_attrs.parquet — one row per (target, context) pair
    loo_pilot_meta.parquet  — one row per token in each processed document
    loo_pilot_params.json   — hyperparameters and runtime stats
    fig/loo_pilot_profiles.pdf — POS × distance heatmaps
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.paths import DATA_DIR, FIG_DIR
from features.predictability import pairwise

ELLIPSE_PATH = DATA_DIR / "ELLIPSE_Final_github.csv"
OUTPUT_DIR = DATA_DIR / "loo_pilot"
MODEL_NAME = "answerdotai/ModernBERT-base"

# POS coarsening: 8 categories
POS_COARSE_MAP = {
    "NOUN": "NOUN", "PROPN": "NOUN",
    "VERB": "VERB", "AUX": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "PRON": "PRON",
    "DET": "DET",
    "ADP": "ADP",
}
POS_ORDER = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "OTHER"]

# Distance bands
BAND_ORDER = ["1", "2-3", "4-7", "8+"]


def get_pos_coarse(pos):
    """Map fine-grained POS to 8 coarse categories."""
    return POS_COARSE_MAP.get(pos, "OTHER")


def get_distance_band(abs_dist):
    """Map absolute distance to band label."""
    if abs_dist == 1:
        return "1"
    elif abs_dist <= 3:
        return "2-3"
    elif abs_dist <= 7:
        return "4-7"
    else:
        return "8+"


def get_token_alignment(tokenizer, doc):
    """Align spaCy tokens with transformer subword tokens.

    Returns:
        token_map: dict mapping spacy_idx -> (subword_start, subword_end)
        trf_tok_ids: numpy array of subword token IDs
    """
    encoding = tokenizer(
        doc.text,
        add_special_tokens=False,
        return_tensors="np",
        return_offsets_mapping=True,
    )
    trf_tok_ids = encoding.input_ids[0]
    offset_mapping = encoding["offset_mapping"][0]

    adjusted_starts = []
    for start, end in offset_mapping:
        token_text = doc.text[start:end]
        stripped_start = start + len(token_text) - len(token_text.lstrip())
        adjusted_starts.append(stripped_start)
    trf_starts = np.array(adjusted_starts)

    spacy_starts = np.array([t.idx for t in doc])

    _, common2spacy, common2trf = np.intersect1d(
        spacy_starts, trf_starts, return_indices=True
    )

    token_map = {}
    for spacy_ind, (st, end) in zip(common2spacy, pairwise(common2trf)):
        token_map[spacy_ind] = (st, end)

    return token_map, trf_tok_ids


# ── compute ──────────────────────────────────────────────────────────────

def cmd_compute(args):
    """Sample documents and compute double-masked LOO attributions."""
    import spacy
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig

    # Load essays and compute proficiency tertiles
    print("Loading ELLIPSE essays...")
    essays_df = pd.read_csv(ELLIPSE_PATH)
    essays_df["prof_tertile"] = pd.qcut(
        essays_df["Overall"], q=3, labels=["low", "mid", "high"]
    )
    print(f"  {len(essays_df)} essays, tertile counts: "
          f"{essays_df['prof_tertile'].value_counts().to_dict()}")

    # Sample documents stratified by proficiency tertile
    rng = np.random.default_rng(args.seed)
    sampled = []
    for tertile in ["low", "mid", "high"]:
        tdf = essays_df[essays_df["prof_tertile"] == tertile]
        n = min(args.n_docs_per_tertile, len(tdf))
        idx = rng.choice(len(tdf), size=n, replace=False)
        sampled.append(tdf.iloc[idx])
    sample_df = pd.concat(sampled).reset_index(drop=True)
    essay_id_list = sample_df["text_id_kaggle"].tolist()
    print(f"Sampled {len(sample_df)} documents: "
          f"{sample_df['prof_tertile'].value_counts().to_dict()}")
    print(f"Essay IDs: {essay_id_list}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.reference_compile = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, config=config)
    model.to(args.device)
    model.eval()

    # Load spaCy
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_lg")

    # Special token IDs
    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    attr_rows = []
    meta_rows = []
    t0 = time.time()
    n_docs = len(sample_df)
    total_tokens_processed = 0
    total_fwd_passes = 0

    for doc_idx, (_, essay_row) in enumerate(sample_df.iterrows()):
        essay_id = essay_row["text_id_kaggle"]
        essay_text = essay_row["full_text"]
        overall_score = essay_row["Overall"]
        prof_tertile = essay_row["prof_tertile"]

        # spaCy parse + alignment
        doc = nlp(essay_text)
        token_map, trf_tok_ids = get_token_alignment(tokenizer, doc)
        n_spacy = len(token_map)

        if n_spacy == 0:
            print(f"  [{doc_idx+1}/{n_docs}] {essay_id}: "
                  f"no aligned tokens, skipping")
            continue

        # Sorted spaCy indices and their subword ranges
        spacy_indices = sorted(token_map.keys())
        subword_ranges = [token_map[si] for si in spacy_indices]

        # Build full model input: [CLS] + trf_tok_ids + [SEP]
        full_ids = np.concatenate([[cls_id], trf_tok_ids, [sep_id]])
        seq_len = len(full_ids)
        base_input = torch.tensor(
            full_ids.astype(np.int64), device=args.device
        ).unsqueeze(0)
        attn_mask = torch.ones_like(base_input)

        # Pre-compute actual token IDs for gather
        actual_ids_t = torch.tensor(
            full_ids.astype(np.int64), device=args.device
        )

        batch_size = args.batch_size if args.batch_size > 0 else n_spacy

        # ── Phase 1: Baseline passes (mask only target i) ──────────
        # L forward passes, batchable.  For each target token i, mask
        # its subwords and read the MLM surprisal at position i.
        mlm_surprisals = torch.zeros(n_spacy, device=args.device)

        for batch_start in range(0, n_spacy, batch_size):
            batch_end = min(batch_start + batch_size, n_spacy)
            bs = batch_end - batch_start

            batch = base_input.repeat(bs, 1)
            batch_attn = attn_mask.repeat(bs, 1)

            for b, i_k in enumerate(range(batch_start, batch_end)):
                sub_s, sub_e = subword_ranges[i_k]
                batch[b, sub_s + 1: sub_e + 1] = mask_id

            with torch.no_grad():
                out = model(input_ids=batch, attention_mask=batch_attn)
                logprobs = torch.log_softmax(out.logits, dim=-1)
                actual_lp = logprobs[
                    torch.arange(bs, device=args.device).unsqueeze(1),
                    torch.arange(seq_len, device=args.device).unsqueeze(0),
                    actual_ids_t.unsqueeze(0),
                ]
                del logprobs, out

            neg_lp = -actual_lp  # (bs, seq_len)

            for b, i_k in enumerate(range(batch_start, batch_end)):
                sub_s, sub_e = subword_ranges[i_k]
                mlm_surprisals[i_k] = neg_lp[
                    b, sub_s + 1: sub_e + 1
                ].sum()

            del neg_lp, actual_lp, batch, batch_attn
            total_fwd_passes += bs

        # ── Phase 2: Pairwise passes (mask BOTH i and j) ──────────
        # Each forward pass masks two tokens; we read MLM surprisal at
        # both masked positions, filling loo_deltas[i,j] AND [j,i].
        # Only unordered pairs {i,j} with i<j needed → L(L-1)/2 passes.
        loo_deltas = torch.zeros(n_spacy, n_spacy, device=args.device)

        # Build list of all unordered pairs
        pairs = [(i, j) for i in range(n_spacy) for j in range(i + 1, n_spacy)]
        n_pairs = len(pairs)

        for pb_start in range(0, n_pairs, batch_size):
            pb_end = min(pb_start + batch_size, n_pairs)
            pair_batch = pairs[pb_start:pb_end]
            bs = len(pair_batch)

            batch = base_input.repeat(bs, 1)
            batch_attn = attn_mask.repeat(bs, 1)

            for b, (i_k, j_k) in enumerate(pair_batch):
                i_sub_s, i_sub_e = subword_ranges[i_k]
                j_sub_s, j_sub_e = subword_ranges[j_k]
                batch[b, i_sub_s + 1: i_sub_e + 1] = mask_id
                batch[b, j_sub_s + 1: j_sub_e + 1] = mask_id

            with torch.no_grad():
                out = model(input_ids=batch, attention_mask=batch_attn)
                logprobs = torch.log_softmax(out.logits, dim=-1)
                actual_lp = logprobs[
                    torch.arange(bs, device=args.device).unsqueeze(1),
                    torch.arange(seq_len, device=args.device).unsqueeze(0),
                    actual_ids_t.unsqueeze(0),
                ]
                del logprobs, out

            neg_lp = -actual_lp  # (bs, seq_len)

            for b, (i_k, j_k) in enumerate(pair_batch):
                i_sub_s, i_sub_e = subword_ranges[i_k]
                j_sub_s, j_sub_e = subword_ranges[j_k]

                # Surprisal at i when both i,j masked
                surp_i = neg_lp[b, i_sub_s + 1: i_sub_e + 1].sum()
                loo_deltas[i_k, j_k] = surp_i - mlm_surprisals[i_k]

                # Surprisal at j when both i,j masked
                surp_j = neg_lp[b, j_sub_s + 1: j_sub_e + 1].sum()
                loo_deltas[j_k, i_k] = surp_j - mlm_surprisals[j_k]

            del neg_lp, actual_lp, batch, batch_attn
            total_fwd_passes += bs

            # Progress within document
            if (pb_start // batch_size) % 100 == 0 and pb_start > 0:
                elapsed = time.time() - t0
                print(f"    {pb_start}/{n_pairs} pairs, "
                      f"{total_fwd_passes} fwd passes, {elapsed:.0f}s")

        # ── Build KWIC contexts ──
        def get_kwic(spacy_idx, n_chars=50):
            tok = doc[spacy_idx]
            left = doc.text[:tok.idx]
            right = doc.text[tok.idx + len(tok.text):]
            return left[-n_chars:], right[:n_chars]

        # ── Save rows ──
        loo_deltas_np = loo_deltas.cpu().numpy()
        mlm_surprisals_np = mlm_surprisals.cpu().numpy()

        for i_k, i_si in enumerate(spacy_indices):
            i_tok = doc[i_si]
            left_ctx, right_ctx = get_kwic(i_si)

            sum_loo_abs = np.abs(loo_deltas_np[i_k]).sum()

            meta_rows.append({
                "doc_idx": doc_idx,
                "essay_id": essay_id,
                "spacy_idx": i_si,
                "token_text": i_tok.text,
                "pos_tag": i_tok.pos_,
                "dep_rel": i_tok.dep_,
                "mlm_surprisal": float(mlm_surprisals_np[i_k]),
                "overall_score": float(overall_score),
                "prof_tertile": str(prof_tertile),
                "left_context": left_ctx,
                "right_context": right_ctx,
                "sum_loo_delta_abs": float(sum_loo_abs),
            })

            for j_k, j_si in enumerate(spacy_indices):
                if j_k == i_k:
                    continue
                j_tok = doc[j_si]
                distance = j_si - i_si
                abs_distance = abs(distance)

                attr_rows.append({
                    "doc_idx": doc_idx,
                    "essay_id": essay_id,
                    "target_spacy_idx": i_si,
                    "target_text": i_tok.text,
                    "target_pos": i_tok.pos_,
                    "target_pos_coarse": get_pos_coarse(i_tok.pos_),
                    "ctx_spacy_idx": j_si,
                    "ctx_text": j_tok.text,
                    "ctx_pos": j_tok.pos_,
                    "ctx_pos_coarse": get_pos_coarse(j_tok.pos_),
                    "distance": distance,
                    "abs_distance": abs_distance,
                    "distance_band": get_distance_band(abs_distance),
                    "loo_delta": float(loo_deltas_np[i_k, j_k]),
                })

        total_tokens_processed += n_spacy
        elapsed = time.time() - t0
        print(f"  [{doc_idx+1}/{n_docs}] {essay_id}: {n_spacy} tokens, "
              f"{n_spacy**2 - n_spacy} pairs, "
              f"{total_fwd_passes} fwd passes, {elapsed:.0f}s")

    elapsed_total = time.time() - t0
    print(f"\nDone: {n_docs} docs, {total_tokens_processed} tokens, "
          f"{total_fwd_passes} forward passes in {elapsed_total:.0f}s")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attrs_df = pd.DataFrame(attr_rows)
    attrs_df.to_parquet(OUTPUT_DIR / "loo_pilot_attrs.parquet", index=False)
    print(f"Saved {len(attrs_df)} attr rows → "
          f"{OUTPUT_DIR / 'loo_pilot_attrs.parquet'}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(OUTPUT_DIR / "loo_pilot_meta.parquet", index=False)
    print(f"Saved {len(meta_df)} meta rows → "
          f"{OUTPUT_DIR / 'loo_pilot_meta.parquet'}")

    params = {
        "method": "double-masked LOO",
        "n_docs_per_tertile": args.n_docs_per_tertile,
        "n_docs_processed": n_docs,
        "n_tokens_processed": total_tokens_processed,
        "n_attr_rows": len(attrs_df),
        "n_forward_passes": total_fwd_passes,
        "essay_ids": essay_id_list,
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "model": MODEL_NAME,
        "elapsed_seconds": round(elapsed_total, 1),
    }
    with open(OUTPUT_DIR / "loo_pilot_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params → {OUTPUT_DIR / 'loo_pilot_params.json'}")


# ── show ─────────────────────────────────────────────────────────────────

def cmd_show(args):
    """Inspect a single token's attribution map."""
    attrs_path = OUTPUT_DIR / "loo_pilot_attrs.parquet"
    meta_path = OUTPUT_DIR / "loo_pilot_meta.parquet"
    if not attrs_path.exists() or not meta_path.exists():
        print(f"ERROR: output files not found in {OUTPUT_DIR}")
        print("Run `python loo_pilot.py compute` first.")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path)

    doc_idx = args.doc_idx
    spacy_idx = args.spacy_idx

    # Find the target token in metadata
    mask = (meta_df["doc_idx"] == doc_idx) & (meta_df["spacy_idx"] == spacy_idx)
    if mask.sum() == 0:
        # Try interpreting doc_idx as essay_id
        mask2 = (meta_df["essay_id"] == str(doc_idx)) & (meta_df["spacy_idx"] == spacy_idx)
        if mask2.sum() > 0:
            mask = mask2
        else:
            print(f"ERROR: (doc_idx={doc_idx}, spacy_idx={spacy_idx}) not found.")
            valid_docs = sorted(meta_df["doc_idx"].unique())
            print(f"Valid doc_idx range: {valid_docs[0]}–{valid_docs[-1]}")
            sys.exit(1)

    tmeta = meta_df[mask].iloc[0]

    print(f"\n=== TOKEN: doc_idx={tmeta['doc_idx']}, spacy_idx={tmeta['spacy_idx']} ===")
    print(f"Token:       {tmeta['token_text']!r:16s} POS: {tmeta['pos_tag']}  "
          f"DEP: {tmeta['dep_rel']}")
    print(f"Essay:       {tmeta['essay_id']}")
    print(f"Proficiency: {tmeta['prof_tertile']} (score={tmeta['overall_score']:.1f})")
    print(f"MLM surprisal:  {tmeta['mlm_surprisal']:.3f} nats")
    print(f"Sum |LOO Δ|:    {tmeta['sum_loo_delta_abs']:.3f}")

    # KWIC
    left = tmeta["left_context"] if isinstance(tmeta["left_context"], str) else ""
    right = tmeta["right_context"] if isinstance(tmeta["right_context"], str) else ""
    print(f"\nKWIC: ...{left}  [ {tmeta['token_text']} ]  {right}...")

    # Attributions for this target
    amask = (
        (attrs_df["doc_idx"] == tmeta["doc_idx"])
        & (attrs_df["target_spacy_idx"] == tmeta["spacy_idx"])
        & (attrs_df["essay_id"] == tmeta["essay_id"])
    )
    tattrs = attrs_df[amask].copy()

    if len(tattrs) == 0:
        print("\nNo context token attributions found.")
        return

    # Sort by |LOO Δ|
    tattrs["abs_delta"] = tattrs["loo_delta"].abs()
    tattrs = tattrs.sort_values("abs_delta", ascending=False)
    total_abs = tattrs["abs_delta"].sum()

    n_show = len(tattrs) if args.all else args.top_n
    show_df = tattrs.head(n_show)

    print(f"\n--- LOO ATTRIBUTIONS (top {n_show} by |Δ|, "
          f"{len(tattrs)} context tokens total) ---")

    print(f" {'#':>3s}  {'DIST':>5s}  {'TOKEN':<14s} "
          f"{'POS':<6s} {'LOO Δ':>8s}  {'EFFECT':>10s}  {'CUMUL%':>6s}")

    cumul = 0.0
    for rank, (_, r) in enumerate(show_df.iterrows(), 1):
        cumul += r["abs_delta"]
        pct = 100 * cumul / total_abs if total_abs > 0 else 0
        dist_str = f"{r['distance']:+d}"
        effect = "helping" if r["loo_delta"] > 0 else "misleading"

        print(f" {rank:>3d}  {dist_str:>5s}  {r['ctx_text']:<14s} "
              f"{r['ctx_pos']:<6s} {r['loo_delta']:>8.4f}  {effect:>10s}"
              f"  {pct:>5.1f}%")


# ── profile ──────────────────────────────────────────────────────────────

def cmd_profile(args):
    """Aggregate attribution profiles by proficiency tertile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    attrs_path = OUTPUT_DIR / "loo_pilot_attrs.parquet"
    meta_path = OUTPUT_DIR / "loo_pilot_meta.parquet"
    if not attrs_path.exists():
        print(f"ERROR: {attrs_path} not found.")
        print("Run `python loo_pilot.py compute` first.")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path) if meta_path.exists() else None

    val_col = "loo_delta"
    method_label = "LOO Δ"

    # Add proficiency tertile to attrs from meta
    if meta_df is not None:
        # Build lookup: (doc_idx, essay_id, spacy_idx) → prof_tertile
        meta_lookup = meta_df.set_index(
            ["doc_idx", "essay_id", "spacy_idx"]
        )["prof_tertile"].to_dict()

        attrs_df = attrs_df.copy()
        attrs_df["target_prof_tertile"] = attrs_df.apply(
            lambda r: meta_lookup.get(
                (r["doc_idx"], r["essay_id"], r["target_spacy_idx"]), None
            ),
            axis=1,
        )
    else:
        print("WARNING: meta file not found, cannot split by proficiency.")
        return

    # Optional target filter: high-surprisal only
    if args.target_filter == "high-surprisal" and meta_df is not None:
        p90 = meta_df["mlm_surprisal"].quantile(0.9)
        high_surp_keys = set(
            zip(
                meta_df[meta_df["mlm_surprisal"] >= p90]["doc_idx"],
                meta_df[meta_df["mlm_surprisal"] >= p90]["essay_id"],
                meta_df[meta_df["mlm_surprisal"] >= p90]["spacy_idx"],
            )
        )
        before = len(attrs_df)
        attrs_df = attrs_df[
            attrs_df.apply(
                lambda r: (r["doc_idx"], r["essay_id"], r["target_spacy_idx"])
                in high_surp_keys,
                axis=1,
            )
        ]
        print(f"Target filter: high-surprisal (≥ {p90:.2f} nats, "
              f"90th pctl) → {len(attrs_df)}/{before} rows")

    working = attrs_df.copy()

    if args.abs:
        working[f"abs_{val_col}"] = working[val_col].abs()
        use_col = f"abs_{val_col}"
    else:
        use_col = val_col

    # Build pivot tables per tertile
    tertiles = ["low", "mid", "high"]
    matrices = {}
    for tertile in tertiles:
        tdf = working[working["target_prof_tertile"] == tertile]
        pivot = tdf.pivot_table(
            values=use_col,
            index="ctx_pos_coarse",
            columns="distance_band",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=POS_ORDER, columns=BAND_ORDER)
        matrices[tertile] = pivot

    # Normalization
    if args.normalize == "row":
        for t in tertiles:
            row_sums = matrices[t].abs().sum(axis=1)
            matrices[t] = matrices[t].div(row_sums, axis=0)
    elif args.normalize == "total":
        for t in tertiles:
            total = matrices[t].abs().sum().sum()
            if total > 0:
                matrices[t] = matrices[t] / total

    # Difference matrix: low minus high
    diff = matrices["low"] - matrices["high"]

    # Print text tables
    label = f"|{val_col}|" if args.abs else val_col
    norm_label = (f" (normalize={args.normalize})"
                  if args.normalize != "none" else "")
    filter_label = (f" [target={args.target_filter}]"
                    if args.target_filter != "all" else "")
    print(f"\n=== {method_label} PROFILES: mean {label}"
          f"{norm_label}{filter_label} ===\n")

    for tertile in tertiles:
        print(f"--- {tertile.upper()} ---")
        print(matrices[tertile].to_string(float_format="{:.4f}".format))
        print()

    print("--- LOW − HIGH (difference) ---")
    print(diff.to_string(float_format="{:.4f}".format))
    print()

    # Cell counts
    print("--- CELL COUNTS (low tertile) ---")
    tdf = working[working["target_prof_tertile"] == "low"]
    counts = tdf.pivot_table(
        values=val_col,
        index="ctx_pos_coarse",
        columns="distance_band",
        aggfunc="count",
    ).reindex(index=POS_ORDER, columns=BAND_ORDER).fillna(0).astype(int)
    print(counts.to_string())
    print()

    # Heatmap PDF
    output_path = (
        Path(args.output) if args.output
        else FIG_DIR / "loo_pilot_profiles.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle(
        f"{method_label} Attribution Profiles (POS × Distance) — "
        f"mean {label}{norm_label}{filter_label}",
        fontsize=13,
    )

    panels = [
        ("low", matrices["low"]),
        ("mid", matrices["mid"]),
        ("high", matrices["high"]),
        ("low − high", diff),
    ]

    for ax, (title, mat) in zip(axes, panels):
        data = mat.values.astype(float)

        vmax = np.nanmax(np.abs(data))
        if vmax == 0 or np.isnan(vmax):
            vmax = 1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(BAND_ORDER)))
        ax.set_xticklabels(BAND_ORDER, fontsize=9)
        ax.set_xlabel("Distance band")

        if ax == axes[0]:
            ax.set_yticks(range(len(POS_ORDER)))
            ax.set_yticklabels(POS_ORDER, fontsize=9)
            ax.set_ylabel("Context POS")

        for yi in range(data.shape[0]):
            for xi in range(data.shape[1]):
                val = data[yi, xi]
                if np.isnan(val):
                    ax.text(xi, yi, "—", ha="center", va="center",
                            fontsize=7, color="gray")
                else:
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(xi, yi, f"{val:.3f}", ha="center",
                            va="center", fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved heatmap → {output_path}")


# ── parser ───────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="LOO attribution pilot for surprisal decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compute
    p_compute = sub.add_parser(
        "compute", help="Sample documents and compute LOO attributions"
    )
    p_compute.add_argument(
        "--n-docs-per-tertile", type=int, default=5,
        help="Documents per proficiency tertile (default: 5)"
    )
    p_compute.add_argument("--seed", type=int, default=42)
    p_compute.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)"
    )
    p_compute.add_argument(
        "--batch-size", type=int, default=32,
        help="LOO batch size per forward pass (0 = all at once, default: 32)"
    )
    # show
    p_show = sub.add_parser(
        "show", help="Inspect a single token's attribution map"
    )
    p_show.add_argument("doc_idx", type=int, help="Document index")
    p_show.add_argument("spacy_idx", type=int, help="spaCy token index")
    p_show.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top attributions to show (default: 10)"
    )
    p_show.add_argument(
        "--all", action="store_true",
        help="Show all context token attributions"
    )

    # profile
    p_profile = sub.add_parser(
        "profile", help="Aggregate attribution profiles by proficiency"
    )
    p_profile.add_argument(
        "--normalize", choices=["none", "row", "total"], default="none",
        help="Normalization method (default: none)"
    )
    p_profile.add_argument(
        "--abs", action="store_true",
        help="Use absolute attribution values"
    )
    p_profile.add_argument(
        "--target-filter", choices=["all", "high-surprisal"], default="all",
        help="Filter targets (default: all)"
    )
    p_profile.add_argument(
        "--output", type=str, default=None,
        help="Output PDF path (default: fig/loo_pilot_profiles.pdf)"
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compute":
        cmd_compute(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "profile":
        cmd_profile(args)
