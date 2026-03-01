"""Calculate linguistic features for ELLIPSE essays and merge with predictability."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import spacy
from spacy.tokens import DocBin
from tqdm.auto import tqdm

from features import calculate_all_features, MiCalculator
from util.paths import DATA_DIR


def main():
    # Load ELLIPSE data
    df = pd.read_csv(
        DATA_DIR / "ELLIPSE_Final_github_w_predictability.csv"
    )
    print(f"Loaded {len(df)} essays")

    # Load pre-built DocBins (created by ingest_essays.py)
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    docbin_dir = DATA_DIR / "ellipse_docbins"

    if not docbin_dir.exists() or not any(docbin_dir.glob("*.spacy")):
        raise FileNotFoundError(
            f"DocBins not found at {docbin_dir}. Run ingest_essays.py --ellipse first."
        )

    print("Loading docs from DocBin files...")
    docs = []
    filenames = []
    for file_path in sorted(docbin_dir.glob("*.spacy")):
        doc_bin = DocBin().from_disk(file_path)
        for doc in doc_bin.get_docs(nlp.vocab):
            docs.append(doc)
            meta = doc.user_data.get("meta", None)
            filenames.append(meta["text_id"] if isinstance(meta, dict) else meta)

    print(f"Loaded {len(docs)} documents")
    assert len(docs) == len(df), f"Expected {len(df)} docs, got {len(docs)}"

    # Load reference data
    print("Loading token frequencies...")
    token_freq_df = pd.read_parquet(DATA_DIR / "slim_pajama_lists" / "3grams.parquet")
    token_freq_df = token_freq_df.groupby("token_2", as_index=False)["count"].sum()
    token_freq = dict(zip(token_freq_df["token_2"], token_freq_df["count"]))
    total_tokens = sum(token_freq.values())
    print(f"Loaded {len(token_freq)} unique tokens, total {total_tokens} tokens")

    print("Loading dependency bigrams...")
    dep_df = pd.read_parquet(DATA_DIR / "slim_pajama_lists" / "depgrams.parquet")
    print(f"Loaded {len(dep_df)} dependency bigrams")
    mi_calculator = MiCalculator(dep_df)

    # Calculate features
    results = []
    for i, doc in tqdm(enumerate(docs), total=len(docs), desc="Calculating features"):
        features = calculate_all_features(doc, token_freq, total_tokens, mi_calculator)
        features["text_id_kaggle"] = filenames[i]
        results.append(features)

    df_features = pd.DataFrame(results)
    print(f"Computed features for {len(df_features)} documents")

    # Merge with scores and predictability
    score_cols = [
        "Overall", "Cohesion", "Syntax", "Vocabulary",
        "Phraseology", "Grammar", "Conventions",
    ]
    predictability_cols = ["mean_loss", "mean_entropy", "var_loss"]
    df_meta = df[["text_id_kaggle", "prompt"] + score_cols + predictability_cols]

    df_merged = pd.merge(df_meta, df_features, on="text_id_kaggle", how="inner")
    print(f"Merged dataframe: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")

    output_path = DATA_DIR / "ellipse_metrics.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # NaN audit
    feature_cols = [
        "mtld", "lexical_density", "log_mean_token_freq", "mean_word_length",
        "words_per_sentence", "mod_per_nom", "mean_dep_distance",
        "amod_mi", "dobj_mi", "advmod_mi",
        "content_word_overlap", "connective_density", "sentence_similarity",
        "mean_loss", "mean_entropy", "var_loss",
    ]
    print("\n=== NaN Audit ===")
    nan_counts = df_merged[feature_cols].isna().sum()
    nan_pct = (nan_counts / len(df_merged) * 100).round(2)
    nan_report = pd.DataFrame({"NaN count": nan_counts, "NaN %": nan_pct})
    print(nan_report[nan_report["NaN count"] > 0].to_string())
    if nan_report["NaN count"].sum() == 0:
        print("No NaN values found.")


if __name__ == "__main__":
    main()
