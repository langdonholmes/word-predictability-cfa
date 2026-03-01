"""Self-contained pilot runner: generates pilot_deltas.npy and pilot_metadata.parquet.

Replicates the logic from pilot.ipynb without requiring the full pipeline
(ELLIPSE_token_predictability.parquet). Runs in two passes:

  Pass 1 — Compute per-token surprisal for the 999-essay stratified sample
  Pass 2 — Extract delta vectors at high-surprisal positions

Usage:
    uv run python src/3-vector-transformations/run_pilot.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.predictability import Predictor, get_centered_window
from util.paths import DATA_DIR

# ── Config ────────────────────────────────────────────────────────────
SAMPLE_PER_TERTILE = 333
SEED = 42
WINDOW_SIZE = 64
BATCH_SIZE = 32
TOP_K = 50  # for expected-embedding computation
THRESHOLD_QUANTILE = 0.90  # percentile within top-tertile

crossloss = torch.nn.CrossEntropyLoss(reduction="none")


# ── Delta extraction (adapted from pilot.ipynb cell 2) ───────────────

def compute_transformation_vectors(
    doc, essay_id, overall_score, prof_tertile,
    predictor, W_E, tokenizer, device,
    surprisal_threshold, target_spacy_indices,
    window_size=WINDOW_SIZE, top_k=TOP_K,
):
    """Compute delta = e_pred - e_obs for high-surprisal positions."""
    token_map, trf_tok_ids = predictor.get_token_alignment(doc)
    if not token_map:
        return []

    seq_len = len(trf_tok_ids)
    effective_window = min(window_size, seq_len)
    mask_id = predictor.mask_id

    sequences_to_process = []
    spacy_indices = []
    mask_positions_list = []
    tok_indices_list = []

    for spacy_idx, (subword_start, subword_end) in token_map.items():
        if target_spacy_indices is not None and spacy_idx not in target_spacy_indices:
            continue

        win_start, win_end = get_centered_window(seq_len, subword_start, effective_window)
        window_len = win_end - win_start
        window_ids = trf_tok_ids[win_start:win_end].copy()

        tok_start_in_window = max(0, subword_start - win_start)
        tok_end_in_window = min(window_len, subword_end - win_start)
        if tok_end_in_window <= tok_start_in_window:
            continue

        tok_indices = np.arange(tok_start_in_window, tok_end_in_window)
        window_ids[tok_indices] = mask_id

        sequences_to_process.append(window_ids)
        spacy_indices.append(spacy_idx)
        mask_positions_list.append(tok_indices + 1)
        actual_start = win_start + tok_start_in_window
        actual_end = win_start + tok_end_in_window
        tok_indices_list.append((actual_start, actual_end))

    if not sequences_to_process:
        return []

    results = []
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0

    for i in range(0, len(sequences_to_process), BATCH_SIZE):
        batch_seqs = sequences_to_process[i:i + BATCH_SIZE]
        batch_spacy = spacy_indices[i:i + BATCH_SIZE]
        batch_mask_pos = mask_positions_list[i:i + BATCH_SIZE]
        batch_tok_inds = tok_indices_list[i:i + BATCH_SIZE]

        full_seqs = [[cls_id] + seq.tolist() + [sep_id] for seq in batch_seqs]
        max_len = max(len(s) for s in full_seqs)

        padded_seqs = []
        attention_masks = []
        for seq in full_seqs:
            pad_len = max_len - len(seq)
            padded_seqs.append(seq + [pad_id] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)

        input_ids = torch.tensor(padded_seqs, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)

        with torch.no_grad():
            outputs = predictor.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        for j, (spacy_idx, mask_pos, (sub_start, sub_end)) in enumerate(
            zip(batch_spacy, batch_mask_pos, batch_tok_inds)
        ):
            actual_ids = trf_tok_ids[sub_start:sub_end]
            if len(actual_ids) == 0:
                continue

            # Mean loss across subwords
            losses = []
            for pos, actual_id in zip(mask_pos, actual_ids):
                if pos >= logits.shape[1]:
                    continue
                pred_logits = logits[j, pos]
                target = torch.tensor([int(actual_id)], device=device)
                loss = crossloss(pred_logits.unsqueeze(0), target).item()
                losses.append(loss)

            if not losses:
                continue
            mean_loss = np.mean(losses)

            if mean_loss < surprisal_threshold:
                continue

            # Delta vector (first subword only)
            pos = mask_pos[0]
            if pos >= logits.shape[1]:
                continue
            pred_logits = logits[j, pos]
            probs = torch.softmax(pred_logits, dim=-1)
            actual_id = int(actual_ids[0])

            top_probs, top_idx = torch.topk(probs, k=top_k)
            top_probs = top_probs / top_probs.sum()
            e_pred = (top_probs.unsqueeze(1) * W_E[top_idx]).sum(dim=0)
            e_obs = W_E[actual_id]
            delta = (e_pred - e_obs).cpu().numpy()

            # Top-3 predictions
            top3_probs, top3_idx = torch.topk(probs, k=3)
            top3_tokens = [tokenizer.decode([tid]) for tid in top3_idx.tolist()]
            top3_probs_list = top3_probs.tolist()

            token = doc[spacy_idx]
            left_start = max(0, spacy_idx - 10)
            right_end = min(len(doc), spacy_idx + 11)
            left_context = doc[left_start:spacy_idx].text
            right_context = doc[spacy_idx + 1:right_end].text

            results.append({
                "essay_id": essay_id,
                "spacy_idx": spacy_idx,
                "token_text": token.text,
                "surprisal": mean_loss,
                "delta": delta,
                "pos_tag": token.pos_,
                "dep_rel": token.dep_,
                "left_context": left_context,
                "right_context": right_context,
                "overall_score": overall_score,
                "prof_tertile": prof_tertile,
                "top3_predicted": top3_tokens,
                "top3_probs": top3_probs_list,
            })

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── Load models ───────────────────────────────────────────────────
    print("=" * 60)
    print("PILOT RUNNER — NMF Delta Vector Extraction")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}")

    model_name = "answerdotai/ModernBERT-base"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    W_E = model.get_input_embeddings().weight.detach()
    print(f"Embedding matrix: {W_E.shape}")

    # Verify tied embeddings
    assert (model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()), \
        "Embeddings are NOT tied!"
    print("Tied embeddings verified")

    predictor = Predictor(
        tokenizer=tokenizer, model=model,
        model_type="masked", batch_size=BATCH_SIZE, device=device,
    )

    print("\nLoading spaCy...")
    nlp = spacy.load("en_core_web_lg")
    print("spaCy loaded")

    # ── Load and sample essays ────────────────────────────────────────
    essays = pd.read_csv(DATA_DIR / "ELLIPSE_Final_github.csv")
    print(f"\nLoaded {len(essays)} essays")

    essays["prof_tertile"] = pd.qcut(
        essays["Overall"], q=3, labels=["low", "mid", "high"]
    )
    print(f"Proficiency distribution:\n{essays['prof_tertile'].value_counts()}")

    sample = essays.groupby("prof_tertile", observed=True).apply(
        lambda g: g.sample(n=min(SAMPLE_PER_TERTILE, len(g)), random_state=SEED),
        include_groups=False,
    ).reset_index(level=0)
    print(f"\nSampled {len(sample)} essays (stratified)")

    # ── Pass 1: Compute per-token surprisal ───────────────────────────
    print("\n" + "=" * 60)
    print("PASS 1 — Computing per-token surprisal")
    print("=" * 60)

    token_records = []
    pass1_errors = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Pass 1 (surprisal)"):
        essay_id = row["text_id_kaggle"]
        text = row["full_text"]
        if pd.isna(text) or text.strip() == "":
            continue
        try:
            doc = nlp(text)
            doc_pred = predictor(doc, window_size=WINDOW_SIZE)
            for token in doc_pred:
                token_records.append({
                    "text_id_kaggle": essay_id,
                    "spacy_idx": token.spacy_idx,
                    "mean_loss": token.mean_loss,
                })
        except Exception as e:
            pass1_errors.append((essay_id, str(e)))

    token_df = pd.DataFrame(token_records)
    print(f"\nPass 1 complete: {len(token_df)} token records")
    if pass1_errors:
        print(f"  Errors: {len(pass1_errors)}")

    # ── Compute threshold ─────────────────────────────────────────────
    # Merge proficiency scores
    token_df = token_df.merge(
        essays[["text_id_kaggle", "Overall"]], on="text_id_kaggle"
    )

    cutoff = essays["Overall"].quantile(0.667)
    reference = token_df[token_df["Overall"] >= cutoff]
    threshold = reference["mean_loss"].quantile(THRESHOLD_QUANTILE)
    print(f"\nProficiency cutoff (top tertile): Overall >= {cutoff:.2f}")
    print(f"Surprisal threshold ({THRESHOLD_QUANTILE:.0%} of top tertile): "
          f"{threshold:.3f} nats")
    print(f"  (Original notebook value: 4.167 nats)")

    # ── Pass 2: Extract delta vectors ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PASS 2 — Extracting delta vectors")
    print("=" * 60)

    # Pre-filter high-surprisal positions
    sample_ids = set(sample["text_id_kaggle"])
    high_surp = token_df[
        (token_df["text_id_kaggle"].isin(sample_ids))
        & (token_df["mean_loss"] >= threshold)
    ]
    high_surp_lookup = (
        high_surp.groupby("text_id_kaggle")["spacy_idx"].apply(set).to_dict()
    )
    print(f"Pre-filtered {len(high_surp)} high-surprisal tokens "
          f"across {len(high_surp_lookup)} essays")

    all_records = []
    pass2_errors = []

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Pass 2 (deltas)"):
        essay_id = row["text_id_kaggle"]
        target_indices = high_surp_lookup.get(essay_id)
        if not target_indices:
            continue

        text = row["full_text"]
        if pd.isna(text) or text.strip() == "":
            continue
        try:
            doc = nlp(text)
            records = compute_transformation_vectors(
                doc=doc,
                essay_id=essay_id,
                overall_score=row["Overall"],
                prof_tertile=row["prof_tertile"],
                predictor=predictor,
                W_E=W_E,
                tokenizer=tokenizer,
                device=device,
                surprisal_threshold=threshold,
                target_spacy_indices=target_indices,
            )
            all_records.extend(records)
        except Exception as e:
            pass2_errors.append((essay_id, str(e)))

    print(f"\nPass 2 complete: {len(all_records)} delta vectors extracted")
    if pass2_errors:
        print(f"  Errors: {len(pass2_errors)}")

    # ── Save ──────────────────────────────────────────────────────────
    deltas = np.array([r["delta"] for r in all_records], dtype=np.float32)
    metadata = pd.DataFrame([
        {k: v for k, v in r.items() if k != "delta"} for r in all_records
    ])

    print(f"\nDelta matrix shape: {deltas.shape}")
    np.save(DATA_DIR / "pilot_deltas.npy", deltas)
    metadata.to_parquet(DATA_DIR / "pilot_metadata.parquet", index=False)
    print(f"Saved pilot_deltas.npy and pilot_metadata.parquet")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
