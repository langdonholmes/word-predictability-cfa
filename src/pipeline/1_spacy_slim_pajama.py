"""Parse SlimPajama texts with spaCy and save as DocBin files."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm.auto import tqdm

from util.paths import DATA_DIR
from util.process_docs import process_dataframe, load_all_docbins


def main():
    df = pd.read_parquet(DATA_DIR / "slim-pajama-test-no-code.parquet")
    print(f"Loaded {len(df)} documents")

    process_dataframe(
        df=df,
        text_col="text",
        metadata_col="meta",
        output_dir=str(DATA_DIR / "slim_pajama_docbins"),
        n_process=32,
        batch_size=512,
    )

    # Verify: load docbins back
    docs = list(
        tqdm(
            load_all_docbins(str(DATA_DIR / "slim_pajama_docbins")),
            desc="Verifying docs",
            total=len(df),
        )
    )
    print(f"Verified {len(docs)} documents")


if __name__ == "__main__":
    main()
