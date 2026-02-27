"""Calculate word predictability metrics for ELLIPSE essays using ModernBERT."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm.auto import tqdm

from features.predictability import Predictor
from util.paths import DATA_DIR


def main():
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        torch.set_float32_matmul_precision("high")

    # Load the dataset
    df = pd.read_csv(DATA_DIR / "ELLIPSE_Final_github.csv")
    print(f"Loaded {len(df)} essays")

    # Load spaCy
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")
    print("spaCy loaded")

    # Load ModernBERT
    model_name = "answerdotai/ModernBERT-base"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    print("ModernBERT loaded")

    # Initialize predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    window_size = 64
    batch_size = 32

    predictor = Predictor(
        tokenizer=tokenizer,
        model=model,
        model_type="masked",
        batch_size=batch_size,
        device=device,
    )
    print(f"Predictor initialized (device={device})")

    # Process each essay
    results = {"mean_loss": [], "mean_prob": [], "mean_entropy": [], "var_loss": []}
    token_records = []  # Per-token data for downstream analyses
    print(f"Processing {len(df)} essays...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating predictability"):
        text = row["full_text"]

        if pd.isna(text) or text.strip() == "":
            results["mean_loss"].append(None)
            results["mean_prob"].append(None)
            results["mean_entropy"].append(None)
            results["var_loss"].append(None)
            continue

        try:
            doc = nlp(text)
            doc_pred = predictor(doc, window_size=window_size)

            results["mean_loss"].append(doc_pred.mean_loss)
            results["mean_prob"].append(
                sum(t.mean_prob for t in doc_pred) / len(doc_pred)
            )
            results["mean_entropy"].append(doc_pred.mean_entropy)
            results["var_loss"].append(doc_pred.var_loss)

            # Collect per-token records
            for token in doc_pred:
                record = token.to_dict()
                record["text_id_kaggle"] = row["text_id_kaggle"]
                token_records.append(record)

        except Exception as e:
            print(f"\nError processing essay {idx} (text_id_kaggle: {row['text_id_kaggle']}): {e}")
            results["mean_loss"].append(None)
            results["mean_prob"].append(None)
            results["mean_entropy"].append(None)
            results["var_loss"].append(None)

    print("\nProcessing complete!")

    # Add results to dataframe and save
    df["mean_loss"] = results["mean_loss"]
    df["mean_prob"] = results["mean_prob"]
    df["mean_entropy"] = results["mean_entropy"]
    df["var_loss"] = results["var_loss"]

    print("Summary statistics for predictability metrics:\n")
    print(df[["mean_loss", "mean_prob", "mean_entropy", "var_loss"]].describe())

    output_path = DATA_DIR / "ELLIPSE_Final_github_w_predictability.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Save per-token data as parquet
    token_df = pd.DataFrame(token_records)
    token_parquet_path = DATA_DIR / "ELLIPSE_token_predictability.parquet"
    token_df.to_parquet(token_parquet_path, index=False)
    print(f"Saved {len(token_df)} token records to: {token_parquet_path}")


if __name__ == "__main__":
    main()
