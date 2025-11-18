import spacy
from spacy.tokens import DocBin
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from typing import Iterator, Tuple


def generate_text_metadata_pairs(
    df: pd.DataFrame, text_col: str, metadata_col: str
) -> Iterator[Tuple[str, str]]:
    """
    Generate pairs of (text, metadata) from dataframe.
    Assumes column names are valid Python attributes (i.e., do not contain spaces).
    """
    for row in df.itertuples():
        yield (getattr(row, text_col), getattr(row, metadata_col))


def process_dataframe(
    df: pd.DataFrame,
    text_col: str,
    metadata_col: str,
    output_dir: str,
    model: str = "en_core_web_lg",
    n_process: int = 32,
    batch_size: int = 512,
):
    """
    Process dataframe using parallel processing and save to multiple DocBin files.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing texts and metadata
    text_col : str
        Name of column containing text to process
    metadata_col : str
        Name of column containing metadata to preserve
    output_dir : str or Path
        Directory to save the DocBin files
    max_docbin_size_mb : float
        Maximum size for each DocBin file in megabytes
    model : str
        Name of spaCy model to use
    n_process : int
        Number of processes to use for parallel processing
    batch_size : int
        Number of documents to process in each batch for nlp.pipe()
    """
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = 5_000_000
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    current_docbin_idx = 0

    # Process using nlp.pipe() with parallel processing and as_tuples=True
    with tqdm(total=len(df), desc="Processing documents") as pbar:
        for i, (doc, meta) in enumerate(
            nlp.pipe(
                generate_text_metadata_pairs(df, text_col, metadata_col),
                as_tuples=True,
                n_process=n_process,
                batch_size=batch_size,
            )
        ):
            # Store metadata
            doc.user_data["meta"] = meta

            # Add to current DocBin and check size
            docs.append(doc)

            # Save every 100,000 docs
            if i % 10_000 == 0 and i > 0:
                output_path = (
                    output_dir / f"processed_docs_{current_docbin_idx:02}.spacy"
                )
                docbin = DocBin(docs=docs, store_user_data=True)
                docbin.to_disk(output_path)
                print(f"\nSaved DocBin {current_docbin_idx} to {output_path}")
                current_docbin_idx += 1
                docs = []

            # Update progress bar
            pbar.update(1)

    # Save final DocBin if it contains any documents
    if len(docs) > 0:
        output_path = output_dir / f"processed_docs_{current_docbin_idx:02}.spacy"
        docbin = DocBin(docs=docs, store_user_data=True)
        docbin.to_disk(output_path)
        print(f"\nSaved final DocBin {current_docbin_idx} to {output_path}")


def load_all_docbins(directory):
    """
    Generator function to load and yield all docs from multiple DocBin files.
    """
    directory = Path(directory)
    nlp = spacy.blank("en")  # Light-weight model just for loading

    for file_path in sorted(directory.glob("*.spacy")):
        doc_bin = DocBin().from_disk(file_path)
        for doc in doc_bin.get_docs(nlp.vocab):
            yield doc


if __name__ == "__main__":
    df = pd.DataFrame(
        {"text": ["Sample text 1", "Sample text 2"], "metadata": ["meta1", "meta2"]}
    )

    process_dataframe(
        df=df,
        text_col="text",
        metadata_col="metadata",
        output_dir="processed_docs",
        n_process=16,  # Use 32 cores
        batch_size=100,  # Adjust based on available RAM
    )
