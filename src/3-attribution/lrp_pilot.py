"""AttnLRP attribution pilot for decomposing token-level surprisal.

Uses LXT (LRP eXplains Transformers) with BERT to compute per-context-token
relevance scores, answering: "which context tokens made the model surprised
at this position?"

AttnLRP (Arras et al., 2025) requires only 1 forward + 1 backward pass per
token — roughly 30x faster than Integrated Gradients. The relevance scores
approximately conserve the target value (surprisal).

This pilot uses bert-base-uncased (not ModernBERT) because LXT has built-in
AttnLRP support for BERT. The model differs from the rest of the pipeline,
but for a pilot exploring method viability this is acceptable.

Subcommands:
    compute  — Sample tokens and compute AttnLRP attributions
    show     — Inspect a single token's attribution map
    profile  — Aggregate attribution profiles by proficiency

Example workflow:
    python lrp_pilot.py compute --n-per-tertile 5
    python lrp_pilot.py show 0
    python lrp_pilot.py profile

Source data (read-only, from pilot.ipynb):
    data/pilot_metadata.parquet  — 55,998 high-surprisal tokens
    data/ELLIPSE_Final_github.csv — essay texts

Outputs (in data/lrp_pilot/):
    lrp_pilot_attrs.parquet — one row per (target, context) pair
    lrp_pilot_meta.parquet  — one row per target token
    lrp_pilot_params.json   — hyperparameters and runtime stats
    fig/lrp_pilot_profiles.pdf — POS × distance heatmaps
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

METADATA_PATH = DATA_DIR / "pilot_metadata.parquet"
ELLIPSE_PATH = DATA_DIR / "ELLIPSE_Final_github.csv"
OUTPUT_DIR = DATA_DIR / "lrp_pilot"
MODEL_NAME = "bert-base-uncased"

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
    from features.predictability import pairwise

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
    """Sample tokens and compute AttnLRP attributions."""
    import spacy
    import torch
    import transformers.models.bert.modeling_bert as modeling_bert
    from lxt.efficient import monkey_patch
    from features.predictability import get_centered_window

    # Patch BERT module for AttnLRP before loading model
    monkey_patch(modeling_bert, verbose=True)

    # Load metadata and sample
    metadata_path = Path(args.metadata) if args.metadata else METADATA_PATH
    print(f"Loading metadata from {metadata_path}...")
    meta = pd.read_parquet(metadata_path)

    rng = np.random.default_rng(args.seed)
    samples = []
    for tertile in ["low", "mid", "high"]:
        tertile_df = meta[meta["prof_tertile"] == tertile]
        n = min(args.n_per_tertile, len(tertile_df))
        idx = rng.choice(len(tertile_df), size=n, replace=False)
        samples.append(tertile_df.iloc[idx])
    sample_df = pd.concat(samples).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} tokens: "
          f"{sample_df['prof_tertile'].value_counts().to_dict()}")

    # Load model (must use modeling_bert.BertForMaskedLM, not AutoModel,
    # because monkey_patch patches the module's classes in-place)
    print(f"Loading {MODEL_NAME}...")
    model = modeling_bert.BertForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer_mod = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_mod.AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load spaCy
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_lg")

    # Load essays
    print("Loading essays...")
    essays_df = pd.read_csv(ELLIPSE_PATH)
    essay_texts = dict(zip(essays_df["text_id_kaggle"], essays_df["full_text"]))

    # Special token IDs
    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Embedding layer for LRP
    embed_fn = model.bert.get_input_embeddings()

    # Cache: essay_id -> (doc, token_map, trf_tok_ids)
    essay_cache = {}

    def get_essay_data(essay_id):
        if essay_id not in essay_cache:
            text = essay_texts[essay_id]
            doc = nlp(text)
            token_map, trf_tok_ids = get_token_alignment(tokenizer, doc)
            essay_cache[essay_id] = (doc, token_map, trf_tok_ids)
        return essay_cache[essay_id]

    # Attribution loop
    mode = "contrastive" if args.contrastive else "surprisal"
    print(f"Attribution mode: {mode}")

    attr_rows = []
    meta_rows = []
    n_skipped = 0
    conservation_ratios = []

    t0 = time.time()
    n_total = len(sample_df)

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

        # Centered window around target
        win_start, win_end = get_centered_window(
            seq_len, subword_start, effective_window
        )
        window_ids = trf_tok_ids[win_start:win_end].copy()
        window_len = win_end - win_start

        # Target subword positions within window
        tok_start_w = max(0, subword_start - win_start)
        tok_end_w = min(window_len, subword_end - win_start)
        if tok_end_w <= tok_start_w:
            n_skipped += 1
            continue

        # Mask all subwords of target token
        window_ids[tok_start_w:tok_end_w] = mask_id

        # Full sequence: [CLS] + window + [SEP]
        full_ids = np.concatenate([[cls_id], window_ids, [sep_id]])

        # Mask position in full sequence (+1 for [CLS]), use first subword
        first_mask_pos = tok_start_w + 1
        actual_first_id = int(trf_tok_ids[subword_start])

        # Build input tensors
        input_ids = torch.from_numpy(
            full_ids.astype(np.int64)
        ).unsqueeze(0).to(args.device)
        attn_mask = torch.ones_like(input_ids)

        # Get input embeddings and enable gradients
        input_embeds = embed_fn(input_ids).detach().requires_grad_(True)

        # Forward pass
        outputs = model(inputs_embeds=input_embeds, attention_mask=attn_mask)
        logits = outputs.logits[0, first_mask_pos]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Record surprisal regardless of backward target
        surprisal_val = -log_probs[actual_first_id].item()

        if args.contrastive:
            # Contrastive: backward with mask where top_pred=1, others=-1/V
            V = log_probs.shape[0]
            top_pred = log_probs.argmax().item()
            mask = torch.full_like(log_probs, -1.0 / V)
            mask[top_pred] = 1.0
            target_scalar = (log_probs * mask).sum()
            target_scalar.backward()
        else:
            # Default: backward from raw logit of the actual token.
            # Using the logit (not -log_prob) preserves LRP conservation,
            # since log_softmax mixes all V logits and breaks decomposability.
            # The logit captures "how much context supports this token"
            # while surprisal is recorded separately as metadata.
            target_scalar = logits[actual_first_id]
            target_scalar.backward()

        # Compute relevance
        relevance = (input_embeds * input_embeds.grad).sum(-1).squeeze(0)

        # Conservation ratio
        target_val = target_scalar.item()
        rel_sum = relevance.sum().item()
        conservation = rel_sum / target_val if abs(target_val) > 1e-8 else float("nan")
        conservation_ratios.append(conservation)

        # Convert to numpy for mapping
        position_rels = relevance.detach().cpu().numpy()

        # Zero out gradients for next iteration
        input_embeds.grad = None

        # Map subword relevances back to spaCy word tokens
        context_records = []
        for ctx_spacy_idx, (ctx_sub_start, ctx_sub_end) in token_map.items():
            if ctx_spacy_idx == spacy_idx:
                continue

            # Context subword positions within window
            ctx_start_w = ctx_sub_start - win_start
            ctx_end_w = ctx_sub_end - win_start
            ctx_start_w = max(0, ctx_start_w)
            ctx_end_w = min(window_len, ctx_end_w)

            if ctx_end_w <= ctx_start_w:
                continue

            # Skip if overlaps with target mask region
            if ctx_start_w < tok_end_w and ctx_end_w > tok_start_w:
                continue

            # Sum relevances for this word (+1 for CLS offset)
            word_rel = position_rels[ctx_start_w + 1: ctx_end_w + 1].sum()

            ctx_token = doc[ctx_spacy_idx]
            distance = ctx_spacy_idx - spacy_idx
            abs_distance = abs(distance)

            context_records.append({
                "target_idx": i,
                "target_essay_id": essay_id,
                "target_text": row["token_text"],
                "target_surprisal": row["surprisal"],
                "target_pos": row["pos_tag"],
                "target_prof_tertile": row["prof_tertile"],
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

        meta_rows.append({
            "target_idx": i,
            "essay_id": essay_id,
            "spacy_idx": spacy_idx,
            "token_text": row["token_text"],
            "surprisal": row["surprisal"],
            "surprisal_bert": surprisal_val,
            "target_scalar": target_val,
            "pos_tag": row["pos_tag"],
            "prof_tertile": row["prof_tertile"],
            "conservation_ratio": conservation,
            "n_context_tokens": len(context_records),
            "left_context": row["left_context"],
            "right_context": row["right_context"],
            "top3_predicted": row["top3_predicted"],
            "top3_probs": row["top3_probs"],
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

    # Conservation stats
    cr = np.array(conservation_ratios)
    cr_finite = cr[np.isfinite(cr)]
    if len(cr_finite) > 0:
        print(f"Conservation ratio: mean={cr_finite.mean():.4f}, "
              f"std={cr_finite.std():.4f}, "
              f"median={np.median(cr_finite):.4f}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attrs_df = pd.DataFrame(attr_rows)
    attrs_df.to_parquet(OUTPUT_DIR / "lrp_pilot_attrs.parquet", index=False)
    print(f"Saved {len(attrs_df)} attr rows → {OUTPUT_DIR / 'lrp_pilot_attrs.parquet'}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(OUTPUT_DIR / "lrp_pilot_meta.parquet", index=False)
    print(f"Saved {len(meta_df)} meta rows → {OUTPUT_DIR / 'lrp_pilot_meta.parquet'}")

    params = {
        "method": "AttnLRP",
        "model": MODEL_NAME,
        "mode": mode,
        "n_per_tertile": args.n_per_tertile,
        "window_size": args.window_size,
        "seed": args.seed,
        "device": args.device,
        "n_tokens_computed": len(meta_rows),
        "n_tokens_skipped": n_skipped,
        "n_attr_rows": len(attrs_df),
        "elapsed_seconds": round(elapsed_total, 1),
        "seconds_per_token": round(elapsed_total / max(1, len(meta_rows)), 3),
        "mean_conservation_ratio": float(cr_finite.mean()) if len(cr_finite) > 0 else None,
        "std_conservation_ratio": float(cr_finite.std()) if len(cr_finite) > 0 else None,
    }
    with open(OUTPUT_DIR / "lrp_pilot_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params → {OUTPUT_DIR / 'lrp_pilot_params.json'}")


# ── show ─────────────────────────────────────────────────────────────────

def cmd_show(args):
    """Inspect a single token's attribution map."""
    attrs_path = OUTPUT_DIR / "lrp_pilot_attrs.parquet"
    meta_path = OUTPUT_DIR / "lrp_pilot_meta.parquet"
    if not attrs_path.exists() or not meta_path.exists():
        print(f"ERROR: output files not found in {OUTPUT_DIR}")
        print("Run `python lrp_pilot.py compute` first.")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path)

    target_idx = args.target_idx
    if target_idx not in meta_df["target_idx"].values:
        valid = sorted(meta_df["target_idx"].unique())
        print(f"ERROR: target_idx {target_idx} not found.")
        print(f"Valid range: {valid[0]}–{valid[-1]} ({len(valid)} tokens)")
        sys.exit(1)

    # Target metadata
    tmeta = meta_df[meta_df["target_idx"] == target_idx].iloc[0]

    # Top-3 predictions
    top3_pred = tmeta["top3_predicted"]
    top3_prob = tmeta["top3_probs"]
    pred_strs = []
    if hasattr(top3_pred, "__len__"):
        for j in range(min(3, len(top3_pred))):
            tok = str(top3_pred[j]).strip()
            prob = float(top3_prob[j])
            pred_strs.append(f"{tok}({prob:.2f})")

    print(f"\n=== TARGET TOKEN #{target_idx} ===")
    print(f"Token:        {tmeta['token_text']!r:16s} POS: {tmeta['pos_tag']}")
    print(f"Essay:        {tmeta['essay_id']}  Position: spacy_idx={tmeta['spacy_idx']}")
    print(f"Proficiency:  {tmeta['prof_tertile']}")
    print(f"Surprisal:    {tmeta['surprisal']:.2f} nats (ModernBERT)")
    print(f"              {tmeta['surprisal_bert']:.2f} nats (BERT)")
    print(f"Logit (target): {tmeta['target_scalar']:.4f}")
    if pred_strs:
        print(f"Predictions:  {' '.join(pred_strs)}")
    print(f"Conservation: {tmeta['conservation_ratio']:.4f}")

    # Attributions for this target
    tattrs = attrs_df[attrs_df["target_idx"] == target_idx].copy()
    if len(tattrs) == 0:
        print("\nNo context token attributions found.")
        return

    # Sort by absolute attribution
    tattrs["abs_attr"] = tattrs["attribution"].abs()
    tattrs = tattrs.sort_values("abs_attr", ascending=False)

    total_abs = tattrs["abs_attr"].sum()

    n_show = len(tattrs) if args.all else args.top_n
    show_df = tattrs.head(n_show)

    print(f"\n--- ATTRIBUTIONS (top {n_show} by |score|, "
          f"{len(tattrs)} context tokens total) ---")
    print(f" {'#':>3s}  {'DIST':>5s}  {'DIR':>5s}  {'TOKEN':<14s} "
          f"{'POS':<6s} {'ATTRIB':>8s}  {'CUMUL%':>6s}")

    cumul = 0.0
    for rank, (_, r) in enumerate(show_df.iterrows(), 1):
        cumul += r["abs_attr"]
        pct = 100 * cumul / total_abs if total_abs > 0 else 0
        direction = "left" if r["distance"] < 0 else "right"
        dist_str = f"{r['distance']:+d}"
        print(f" {rank:>3d}  {dist_str:>5s}  {direction:>5s}  "
              f"{r['ctx_text']:<14s} {r['ctx_pos']:<6s} "
              f"{r['attribution']:>8.4f}  {pct:>5.1f}%")

    # KWIC
    print(f"\n--- KWIC ---")
    left = tmeta["left_context"] if isinstance(tmeta["left_context"], str) else ""
    right = tmeta["right_context"] if isinstance(tmeta["right_context"], str) else ""
    token = tmeta["token_text"]
    print(f"...{left[-50:]}  [ {token} ]  {right[:50]}...")


# ── profile ──────────────────────────────────────────────────────────────

def cmd_profile(args):
    """Aggregate attribution profiles by proficiency tertile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    attrs_path = OUTPUT_DIR / "lrp_pilot_attrs.parquet"
    if not attrs_path.exists():
        print(f"ERROR: {attrs_path} not found.")
        print("Run `python lrp_pilot.py compute` first.")
        sys.exit(1)

    attrs_df = pd.read_parquet(attrs_path)

    # Use absolute attributions if requested
    val_col = "attribution"
    if args.abs:
        attrs_df = attrs_df.copy()
        attrs_df["abs_attribution"] = attrs_df["attribution"].abs()
        val_col = "abs_attribution"

    # Build pivot tables per tertile
    tertiles = ["low", "mid", "high"]
    matrices = {}
    for tertile in tertiles:
        tdf = attrs_df[attrs_df["target_prof_tertile"] == tertile]
        pivot = tdf.pivot_table(
            values=val_col,
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
    norm_label = f" (normalize={args.normalize})" if args.normalize != "none" else ""
    print(f"\n=== LRP ATTRIBUTION PROFILES: mean {label}{norm_label} ===\n")

    for tertile in tertiles:
        print(f"--- {tertile.upper()} ---")
        print(matrices[tertile].to_string(float_format="{:.4f}".format))
        print()

    print("--- LOW − HIGH (difference) ---")
    print(diff.to_string(float_format="{:.4f}".format))
    print()

    # Cell counts for reference
    print("--- CELL COUNTS (low tertile) ---")
    tdf = attrs_df[attrs_df["target_prof_tertile"] == "low"]
    counts = tdf.pivot_table(
        values="attribution",
        index="ctx_pos_coarse",
        columns="distance_band",
        aggfunc="count",
    ).reindex(index=POS_ORDER, columns=BAND_ORDER).fillna(0).astype(int)
    print(counts.to_string())
    print()

    # Heatmap PDF
    output_path = Path(args.output) if args.output else FIG_DIR / "lrp_pilot_profiles.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle(
        f"AttnLRP Attribution Profiles (POS × Distance) — mean {label}{norm_label}",
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
        if vmax == 0:
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
                    ax.text(xi, yi, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved heatmap → {output_path}")


# ── parser ───────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="AttnLRP pilot for surprisal decomposition (BERT + LXT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compute
    p_compute = sub.add_parser(
        "compute", help="Sample tokens and compute AttnLRP attributions"
    )
    p_compute.add_argument(
        "--contrastive", action="store_true",
        help="Use contrastive target (top-pred vs others) instead of raw surprisal"
    )
    p_compute.add_argument(
        "--n-per-tertile", type=int, default=334,
        help="Tokens to sample per proficiency tertile (default: 334)"
    )
    p_compute.add_argument(
        "--window-size", type=int, default=64,
        help="Context window in subword tokens (default: 64, max ~510 for BERT)"
    )
    p_compute.add_argument("--seed", type=int, default=42)
    p_compute.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)"
    )
    p_compute.add_argument(
        "--metadata", type=str, default=None,
        help="Custom metadata parquet path (default: pilot_metadata.parquet)"
    )

    # show
    p_show = sub.add_parser(
        "show", help="Inspect a single token's attribution map"
    )
    p_show.add_argument(
        "target_idx", type=int, help="Target token index to inspect"
    )
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
        "--output", type=str, default=None,
        help="Output PDF path (default: fig/lrp_pilot_profiles.pdf)"
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
