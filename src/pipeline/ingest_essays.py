"""Ingest TOEFL 11 and ELLIPSE essays into spaCy DocBins."""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from util.paths import DATA_DIR
from util.process_docs import process_dataframe


def ingest_toefl11(force: bool = False):
    """Read TOEFL 11 essays from zip archive and save as DocBins."""
    docbin_dir = DATA_DIR / "toefl11_docbins"

    if docbin_dir.exists() and any(docbin_dir.glob("*.spacy")) and not force:
        print(f"TOEFL 11 DocBins already exist at {docbin_dir}, skipping (use --force to regenerate)")
        return

    # Load index
    index_df = pd.read_csv(DATA_DIR / "index-training-dev.csv")
    print(f"Loaded TOEFL 11 index: {len(index_df)} entries")

    # Map bare filenames to zip paths
    zip_path = DATA_DIR / "toefl_11_txt.zip"
    with zipfile.ZipFile(zip_path) as z:
        file_paths = {}
        for name in z.namelist():
            if name.endswith(".txt"):
                bare_name = name.split("/")[-1]
                file_paths[bare_name] = name

        # Check for missing files
        missing = set(index_df["Text"]) - set(file_paths.keys())
        if missing:
            print(f"WARNING: {len(missing)} index entries have no matching file in zip")
            index_df = index_df[~index_df["Text"].isin(missing)]

        # Read texts
        texts = []
        for filename in index_df["Text"]:
            raw = z.read(file_paths[filename])
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
                print(f"WARNING: {filename} fell back to latin-1 decoding")
            texts.append(text)

    index_df["full_text"] = texts

    # Drop empty essays
    empty_mask = index_df["full_text"].str.strip() == ""
    if empty_mask.any():
        print(f"WARNING: Dropping {empty_mask.sum()} empty essays")
        index_df = index_df[~empty_mask]

    # Build metadata dict column
    index_df["meta"] = index_df.apply(
        lambda row: {
            "corpus": "toefl11",
            "text_id": row["Text"],
            "l1": row["L1"],
            "prompt": row["Prompt"],
            "level": row["Level"],
        },
        axis=1,
    )

    print(f"Processing {len(index_df)} TOEFL 11 essays...")
    if docbin_dir.exists():
        shutil.rmtree(docbin_dir)

    process_dataframe(
        df=index_df,
        text_col="full_text",
        metadata_col="meta",
        output_dir=str(docbin_dir),
        model="en_core_web_lg",
        n_process=32,
        batch_size=512,
    )
    print(f"TOEFL 11 DocBins saved to {docbin_dir}")


def ingest_ellipse(force: bool = False):
    """Read ELLIPSE essays from CSV and save as DocBins."""
    docbin_dir = DATA_DIR / "ellipse_docbins"

    if docbin_dir.exists() and any(docbin_dir.glob("*.spacy")) and not force:
        print(f"ELLIPSE DocBins already exist at {docbin_dir}, skipping (use --force to regenerate)")
        return

    df = pd.read_csv(DATA_DIR / "ELLIPSE_Final_github.csv")
    print(f"Loaded ELLIPSE: {len(df)} essays")

    df = df.dropna(subset=["full_text"])

    score_cols = [
        "Overall", "Cohesion", "Syntax", "Vocabulary",
        "Phraseology", "Grammar", "Conventions",
    ]
    df["meta"] = df.apply(
        lambda row: {
            "corpus": "ellipse",
            "text_id": row["text_id_kaggle"],
            "prompt": row["prompt"],
            **{col.lower(): float(row[col]) for col in score_cols},
        },
        axis=1,
    )

    print(f"Processing {len(df)} ELLIPSE essays...")
    if docbin_dir.exists():
        shutil.rmtree(docbin_dir)

    process_dataframe(
        df=df,
        text_col="full_text",
        metadata_col="meta",
        output_dir=str(docbin_dir),
        model="en_core_web_lg",
        n_process=32,
        batch_size=512,
    )
    print(f"ELLIPSE DocBins saved to {docbin_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ingest essay corpora into spaCy DocBins")
    parser.add_argument("--toefl", action="store_true", help="Ingest TOEFL 11 only")
    parser.add_argument("--ellipse", action="store_true", help="Ingest ELLIPSE only")
    parser.add_argument("--all", action="store_true", help="Ingest all corpora")
    parser.add_argument("--force", action="store_true", help="Regenerate even if DocBins exist")
    args = parser.parse_args()

    if not (args.toefl or args.ellipse or args.all):
        parser.print_help()
        sys.exit(1)

    if args.toefl or args.all:
        ingest_toefl11(force=args.force)
    if args.ellipse or args.all:
        ingest_ellipse(force=args.force)


if __name__ == "__main__":
    main()
