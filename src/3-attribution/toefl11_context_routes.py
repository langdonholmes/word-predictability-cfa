"""Context-route classification pilot on TOEFL11.

Tests whether decomposing surprisal into context-route features via AttnLRP
can meaningfully predict proficiency (high/medium/low) or L1 (11 languages).

TOEFL11 has wider proficiency separation than ELLIPSE and L1 labels, making
it a better test case for the RQ of whether context routes differ across
proficiency levels.

Subcommands:
    compute  — Sample essays/tokens, run AttnLRP, save attributions
    predict  — Extract features, run classification, print results

Example:
    python toefl11_context_routes.py compute --device cuda
    python toefl11_context_routes.py predict
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.paths import DATA_DIR

# Reuse helpers from lrp_pilot
from lrp_pilot import (
    get_token_alignment,
    get_pos_coarse,
    get_distance_band,
    POS_COARSE_MAP,
    POS_ORDER,
    BAND_ORDER,
)

INDEX_PATH = DATA_DIR / "index-training-dev.csv"
DOCBIN_DIR = DATA_DIR / "toefl11_docbins"
OUTPUT_DIR = DATA_DIR / "toefl11_context_routes"


# ── compute ──────────────────────────────────────────────────────────────

def cmd_compute(args):
    """Sample essays and tokens, run AttnLRP, save attribution data."""
    import spacy
    import torch
    from spacy.tokens import DocBin
    from transformers import AutoTokenizer, AutoConfig
    from features.predictability import get_centered_window
    from modernbert_lrp import monkey_patch_modernbert
    import transformers.models.modernbert.modeling_modernbert as modeling_mod

    rng = np.random.default_rng(args.seed)

    # ── Load metadata ────────────────────────────────────────────────
    print(f"Loading metadata from {INDEX_PATH}...")
    index_df = pd.read_csv(INDEX_PATH)
    print(f"  {len(index_df)} essays: "
          f"{index_df['Level'].value_counts().to_dict()}")

    # ── Load docbins ─────────────────────────────────────────────────
    print(f"Loading docbins from {DOCBIN_DIR}...")
    nlp = spacy.load("en_core_web_lg")
    docs_by_id = {}
    for spacy_path in sorted(DOCBIN_DIR.glob("*.spacy")):
        db = DocBin().from_disk(spacy_path)
        for doc in db.get_docs(nlp.vocab):
            text_id = doc.user_data["meta"]["text_id"]
            docs_by_id[text_id] = doc
    print(f"  Loaded {len(docs_by_id)} docs")

    # ── Sample essays: 100 per level, stratified by L1 ──────────────
    n_per_level = args.n_per_level
    sampled_rows = []
    for level in ["low", "medium", "high"]:
        level_df = index_df[index_df["Level"] == level]
        # Ensure all sampled essays have a matching doc
        level_df = level_df[level_df["Text"].isin(docs_by_id)]
        n = min(n_per_level, len(level_df))

        # Stratified by L1: proportional allocation
        l1_counts = level_df["L1"].value_counts()
        l1_fracs = l1_counts / l1_counts.sum()
        l1_targets = (l1_fracs * n).round().astype(int)
        # Fix rounding to hit exactly n
        diff = n - l1_targets.sum()
        if diff != 0:
            biggest = l1_targets.idxmax()
            l1_targets[biggest] += diff

        level_samples = []
        for l1, target_n in l1_targets.items():
            l1_pool = level_df[level_df["L1"] == l1]
            k = min(target_n, len(l1_pool))
            idx = rng.choice(len(l1_pool), size=k, replace=False)
            level_samples.append(l1_pool.iloc[idx])

        sampled_rows.append(pd.concat(level_samples))
        print(f"  {level}: sampled {len(sampled_rows[-1])} essays")

    sample_essays = pd.concat(sampled_rows).reset_index(drop=True)
    print(f"Total sampled essays: {len(sample_essays)}")

    # ── Patch and load model ─────────────────────────────────────────
    monkey_patch_modernbert(modeling_mod, verbose=True)
    ModelClass = modeling_mod.ModernBertForMaskedLM

    model_name = "answerdotai/ModernBERT-base"
    print(f"Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    model = ModelClass.from_pretrained(model_name, config=config)
    model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    embed_fn = model.model.embeddings.tok_embeddings

    # ── Attribution loop ─────────────────────────────────────────────
    attr_rows = []
    meta_rows = []
    n_skipped_tokens = 0
    n_skipped_essays = 0
    conservation_ratios = []
    target_idx_counter = 0

    t0 = time.time()
    n_essays = len(sample_essays)

    for essay_i, (_, essay_row) in enumerate(sample_essays.iterrows()):
        text_id = essay_row["Text"]
        level = essay_row["Level"]
        l1 = essay_row["L1"]
        prompt = essay_row["Prompt"]

        doc = docs_by_id.get(text_id)
        if doc is None:
            n_skipped_essays += 1
            continue

        # Align tokens
        try:
            token_map, trf_tok_ids = get_token_alignment(tokenizer, doc)
        except Exception:
            n_skipped_essays += 1
            continue

        # Eligible tokens: alpha, not PUNCT/SPACE, not first/last
        eligible = []
        for spacy_idx in token_map:
            tok = doc[spacy_idx]
            if tok.pos_ in ("PUNCT", "SPACE"):
                continue
            if not tok.text.isalpha():
                continue
            if spacy_idx == 0 or spacy_idx == len(doc) - 1:
                continue
            eligible.append(spacy_idx)

        if len(eligible) == 0:
            n_skipped_essays += 1
            continue

        # Sample ~60 tokens (or all if fewer)
        n_sample = min(args.tokens_per_essay, len(eligible))
        chosen = rng.choice(eligible, size=n_sample, replace=False)

        seq_len = len(trf_tok_ids)
        effective_window = min(args.window_size, seq_len)

        for spacy_idx in chosen:
            spacy_idx = int(spacy_idx)
            if spacy_idx not in token_map:
                n_skipped_tokens += 1
                continue

            subword_start, subword_end = token_map[spacy_idx]

            # Centered window
            win_start, win_end = get_centered_window(
                seq_len, subword_start, effective_window
            )
            window_ids = trf_tok_ids[win_start:win_end].copy()
            window_len = win_end - win_start

            # Target subword positions within window
            tok_start_w = max(0, subword_start - win_start)
            tok_end_w = min(window_len, subword_end - win_start)
            if tok_end_w <= tok_start_w:
                n_skipped_tokens += 1
                continue

            # Mask target
            window_ids[tok_start_w:tok_end_w] = mask_id

            # [CLS] + window + [SEP]
            full_ids = np.concatenate([[cls_id], window_ids, [sep_id]])
            first_mask_pos = tok_start_w + 1
            actual_first_id = int(trf_tok_ids[subword_start])

            input_ids = torch.from_numpy(
                full_ids.astype(np.int64)
            ).unsqueeze(0).to(args.device)
            attn_mask = torch.ones_like(input_ids)

            input_embeds = embed_fn(input_ids).detach().requires_grad_(True)

            # Forward + backward from raw logit
            outputs = model(inputs_embeds=input_embeds, attention_mask=attn_mask)
            logits = outputs.logits[0, first_mask_pos]
            log_probs = torch.log_softmax(logits, dim=-1)

            surprisal_val = -log_probs[actual_first_id].item()

            target_scalar = logits[actual_first_id]
            target_scalar.backward()

            relevance = (input_embeds * input_embeds.grad).sum(-1).squeeze(0)

            # Conservation
            target_val = target_scalar.item()
            rel_sum = relevance.sum().item()
            conservation = rel_sum / target_val if abs(target_val) > 1e-8 else float("nan")
            conservation_ratios.append(conservation)

            position_rels = relevance.detach().cpu().numpy()
            input_embeds.grad = None

            target_token = doc[spacy_idx]
            pos_tag = target_token.pos_

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
                    "target_idx": target_idx_counter,
                    "essay_id": text_id,
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
                "target_idx": target_idx_counter,
                "essay_id": text_id,
                "spacy_idx": spacy_idx,
                "token_text": target_token.text,
                "pos_tag": pos_tag,
                "surprisal": surprisal_val,
                "target_scalar": target_val,
                "conservation_ratio": conservation,
                "n_context_tokens": len(context_records),
                "level": level,
                "l1": l1,
                "prompt": prompt,
            })

            target_idx_counter += 1

        if (essay_i + 1) % 10 == 0 or essay_i == 0:
            elapsed = time.time() - t0
            rate = elapsed / (essay_i + 1)
            remaining = rate * (n_essays - essay_i - 1)
            print(f"  [{essay_i+1}/{n_essays}] {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining | "
                  f"{target_idx_counter} tokens so far")

    elapsed_total = time.time() - t0
    print(f"\nDone: {len(meta_rows)} tokens from "
          f"{n_essays - n_skipped_essays} essays in {elapsed_total:.0f}s")
    if n_skipped_essays:
        print(f"Skipped essays: {n_skipped_essays}")
    if n_skipped_tokens:
        print(f"Skipped tokens: {n_skipped_tokens}")

    # Conservation stats
    cr = np.array(conservation_ratios)
    cr_finite = cr[np.isfinite(cr)]
    if len(cr_finite) > 0:
        print(f"Conservation ratio: mean={cr_finite.mean():.4f}, "
              f"std={cr_finite.std():.4f}, "
              f"median={np.median(cr_finite):.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attrs_df = pd.DataFrame(attr_rows)
    attrs_df.to_parquet(OUTPUT_DIR / "toefl11_attrs.parquet", index=False)
    print(f"Saved {len(attrs_df)} attr rows → {OUTPUT_DIR / 'toefl11_attrs.parquet'}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(OUTPUT_DIR / "toefl11_meta.parquet", index=False)
    print(f"Saved {len(meta_df)} meta rows → {OUTPUT_DIR / 'toefl11_meta.parquet'}")

    params = {
        "method": "AttnLRP",
        "model": "answerdotai/ModernBERT-base",
        "n_per_level": args.n_per_level,
        "tokens_per_essay": args.tokens_per_essay,
        "window_size": args.window_size,
        "seed": args.seed,
        "device": args.device,
        "n_essays_sampled": n_essays - n_skipped_essays,
        "n_essays_skipped": n_skipped_essays,
        "n_tokens_computed": len(meta_rows),
        "n_tokens_skipped": n_skipped_tokens,
        "n_attr_rows": len(attrs_df),
        "elapsed_seconds": round(elapsed_total, 1),
        "mean_conservation_ratio": float(cr_finite.mean()) if len(cr_finite) > 0 else None,
    }
    with open(OUTPUT_DIR / "toefl11_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params → {OUTPUT_DIR / 'toefl11_params.json'}")


# ── predict ──────────────────────────────────────────────────────────────

def cmd_predict(args):
    """Extract features from attributions and run classification."""
    from scipy.stats import entropy as scipy_entropy
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import LabelEncoder

    attrs_path = OUTPUT_DIR / "toefl11_attrs.parquet"
    meta_path = OUTPUT_DIR / "toefl11_meta.parquet"

    if not attrs_path.exists() or not meta_path.exists():
        print(f"ERROR: output files not found in {OUTPUT_DIR}")
        print("Run `python toefl11_context_routes.py compute` first.")
        sys.exit(1)

    print("Loading attribution data...")
    attrs_df = pd.read_parquet(attrs_path)
    meta_df = pd.read_parquet(meta_path)
    print(f"  {len(meta_df)} target tokens, {len(attrs_df)} attr rows")

    # ── Per-token features ───────────────────────────────────────────
    print("Extracting per-token features...")

    token_features = []
    for target_idx, group in attrs_df.groupby("target_idx"):
        tmeta = meta_df[meta_df["target_idx"] == target_idx]
        if len(tmeta) == 0:
            continue
        tmeta = tmeta.iloc[0]

        abs_attr = group["attribution"].abs()
        total_abs = abs_attr.sum()

        if total_abs < 1e-12:
            # Degenerate case: skip
            continue

        # Normalized |attribution| distribution
        p = abs_attr.values / total_abs

        # 1. surprisal
        surprisal = tmeta["surprisal"]

        # 2. attr_entropy: Shannon entropy of |attribution| distribution
        attr_entropy = scipy_entropy(p)

        # 3. weighted_mean_dist: |attr|-weighted mean absolute distance
        weighted_mean_dist = (p * group["abs_distance"].values).sum()

        # 4. top3_concentration: share of |attr| in top 3 tokens
        top3 = np.sort(abs_attr.values)[-3:]
        top3_concentration = top3.sum() / total_abs

        # 5. left_share: share from preceding context (distance < 0)
        left_mask = group["distance"].values < 0
        left_share = abs_attr.values[left_mask].sum() / total_abs

        # 6-8. POS category shares
        pos_coarse = group["ctx_pos_coarse"].values
        noun_mask = (pos_coarse == "NOUN")
        verb_mask = (pos_coarse == "VERB")
        func_mask = np.isin(pos_coarse, ["DET", "ADP", "PRON"])
        noun_ctx = abs_attr.values[noun_mask].sum() / total_abs
        verb_ctx = abs_attr.values[verb_mask].sum() / total_abs
        func_ctx = abs_attr.values[func_mask].sum() / total_abs

        # 9-12. Distance band shares
        bands = group["distance_band"].values
        band_1 = abs_attr.values[bands == "1"].sum() / total_abs
        band_2_3 = abs_attr.values[bands == "2-3"].sum() / total_abs
        band_4_7 = abs_attr.values[bands == "4-7"].sum() / total_abs
        band_8plus = abs_attr.values[bands == "8+"].sum() / total_abs

        token_features.append({
            "target_idx": target_idx,
            "essay_id": tmeta["essay_id"],
            "level": tmeta["level"],
            "l1": tmeta["l1"],
            "surprisal": surprisal,
            "attr_entropy": attr_entropy,
            "weighted_mean_dist": weighted_mean_dist,
            "top3_concentration": top3_concentration,
            "left_share": left_share,
            "noun_ctx": noun_ctx,
            "verb_ctx": verb_ctx,
            "func_ctx": func_ctx,
            "band_1": band_1,
            "band_2_3": band_2_3,
            "band_4_7": band_4_7,
            "band_8plus": band_8plus,
        })

    token_df = pd.DataFrame(token_features)
    print(f"  {len(token_df)} tokens with features")

    # ── Aggregate per essay ──────────────────────────────────────────
    feature_cols = [
        "surprisal", "attr_entropy", "weighted_mean_dist",
        "top3_concentration", "left_share",
        "noun_ctx", "verb_ctx", "func_ctx",
        "band_1", "band_2_3", "band_4_7", "band_8plus",
    ]

    essay_df = token_df.groupby("essay_id").agg(
        level=("level", "first"),
        l1=("l1", "first"),
        n_tokens=("target_idx", "count"),
        **{col: (col, "mean") for col in feature_cols},
    ).reset_index()
    print(f"  {len(essay_df)} essays with aggregated features")
    print(f"  Level distribution: {essay_df['level'].value_counts().to_dict()}")
    print(f"  Tokens/essay: mean={essay_df['n_tokens'].mean():.1f}, "
          f"min={essay_df['n_tokens'].min()}, max={essay_df['n_tokens'].max()}")

    # ── Classification ───────────────────────────────────────────────
    surprisal_only = ["surprisal"]
    all_features = feature_cols

    tasks = {
        "proficiency": ("level", essay_df["level"].values),
        "L1": ("l1", essay_df["l1"].values),
    }

    feature_sets = {
        "surprisal-only": surprisal_only,
        "all-12-features": all_features,
    }

    models = {
        "LogReg": lambda: LogisticRegression(
            max_iter=1000, solver="lbfgs",
        ),
        "RF": lambda: RandomForestClassifier(
            n_estimators=200, random_state=args.seed, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    print("\n" + "=" * 75)
    print("CLASSIFICATION RESULTS (5-fold stratified CV)")
    print("=" * 75)

    results = []
    for task_name, (label_col, labels) in tasks.items():
        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)

        # Chance baseline
        class_counts = np.bincount(y)
        majority_pct = 100 * class_counts.max() / len(y)
        chance_pct = 100 / n_classes

        print(f"\n--- {task_name.upper()} ({n_classes} classes) ---")
        print(f"  Majority baseline: {majority_pct:.1f}%")
        print(f"  Uniform chance:    {chance_pct:.1f}%")
        print(f"  {'Model':<8s} {'Features':<18s} {'Accuracy':>10s} {'Macro-F1':>10s}")

        for feat_name, feat_list in feature_sets.items():
            X = essay_df[feat_list].values

            for model_name, model_fn in models.items():
                scoring = {"accuracy": "accuracy", "f1": "f1_macro"}
                cv_results = cross_validate(
                    model_fn(), X, y, cv=cv, scoring=scoring,
                )
                acc = cv_results["test_accuracy"].mean()
                f1 = cv_results["test_f1"].mean()
                acc_std = cv_results["test_accuracy"].std()
                f1_std = cv_results["test_f1"].std()

                print(f"  {model_name:<8s} {feat_name:<18s} "
                      f"{acc*100:>5.1f}±{acc_std*100:.1f}%  "
                      f"{f1*100:>5.1f}±{f1_std*100:.1f}%")

                results.append({
                    "task": task_name,
                    "features": feat_name,
                    "model": model_name,
                    "accuracy": acc,
                    "accuracy_std": acc_std,
                    "f1_macro": f1,
                    "f1_std": f1_std,
                })

    # ── Feature importances (RF, all features) ───────────────────────
    print("\n--- FEATURE IMPORTANCES (RandomForest, all 12 features) ---")
    for task_name, (label_col, labels) in tasks.items():
        le = LabelEncoder()
        y = le.fit_transform(labels)
        X = essay_df[all_features].values

        rf = RandomForestClassifier(
            n_estimators=200, random_state=args.seed, n_jobs=-1,
        )
        rf.fit(X, y)

        importances = rf.feature_importances_
        order = np.argsort(importances)[::-1]

        print(f"\n  {task_name}:")
        for rank, idx in enumerate(order, 1):
            print(f"    {rank:2d}. {all_features[idx]:<22s} {importances[idx]:.4f}")

    # ── Summary ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    for task_name in ["proficiency", "L1"]:
        task_results = results_df[results_df["task"] == task_name]
        surp = task_results[task_results["features"] == "surprisal-only"]
        full = task_results[task_results["features"] == "all-12-features"]

        best_surp_acc = surp["accuracy"].max()
        best_full_acc = full["accuracy"].max()
        best_surp_f1 = surp["f1_macro"].max()
        best_full_f1 = full["f1_macro"].max()

        print(f"\n  {task_name}:")
        print(f"    Best surprisal-only:  acc={best_surp_acc*100:.1f}%  "
              f"F1={best_surp_f1*100:.1f}%")
        print(f"    Best all-12-features: acc={best_full_acc*100:.1f}%  "
              f"F1={best_full_f1*100:.1f}%")
        print(f"    Improvement:          acc=+{(best_full_acc-best_surp_acc)*100:.1f}pp  "
              f"F1=+{(best_full_f1-best_surp_f1)*100:.1f}pp")


# ── parser ───────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="TOEFL11 context-route classification pilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compute
    p_compute = sub.add_parser(
        "compute", help="Sample essays/tokens and compute AttnLRP attributions",
    )
    p_compute.add_argument(
        "--n-per-level", type=int, default=100,
        help="Essays to sample per proficiency level (default: 100)",
    )
    p_compute.add_argument(
        "--tokens-per-essay", type=int, default=60,
        help="Alpha tokens to sample per essay (default: 60)",
    )
    p_compute.add_argument(
        "--window-size", type=int, default=64,
        help="Context window in subword tokens (default: 64)",
    )
    p_compute.add_argument("--seed", type=int, default=42)
    p_compute.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )

    # predict
    p_predict = sub.add_parser(
        "predict", help="Extract features and run classification",
    )
    p_predict.add_argument("--seed", type=int, default=42)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compute":
        cmd_compute(args)
    elif args.command == "predict":
        cmd_predict(args)
