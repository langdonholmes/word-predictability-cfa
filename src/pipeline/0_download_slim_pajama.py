"""Download SlimPajama test split and save a code-free subset to parquet."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from util.paths import DATA_DIR


def main():
    # Test split is 500M tokens
    ds_streamed = load_dataset(
        "cerebras/SlimPajama-627B", split="test", streaming=True
    )

    data = [example for example in tqdm(ds_streamed)]
    df = pd.DataFrame(data)

    # Parse metadata
    meta = pd.json_normalize(df.meta)
    df.meta = meta["redpajama_set_name"]

    print(f"Tokens: {df.text.str.split().str.len().sum():,}")

    # Save version without code to disk
    df_no_code = df[
        df.meta.isin(
            [
                "RedPajamaC4",
                "RedPajamaCommonCrawl",
                "RedPajamaBook",
                "RedPajamaWikipedia",
            ]
        )
    ]
    print(f"Documents (no code): {len(df_no_code):,}")

    output_path = DATA_DIR / "slim-pajama-test-no-code.parquet"
    df_no_code.to_parquet(output_path, compression="zstd")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
