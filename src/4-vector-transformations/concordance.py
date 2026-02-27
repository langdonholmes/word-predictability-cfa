"""CLI concordancer for exploring clusters of prediction-error vectors.

This tool supports iterative exploration of transformation vectors extracted
from ModernBERT predictions on L2 learner essays (ELLIPSE corpus). Each
vector represents the difference (delta = e_pred - e_obs) between the
model's expected embedding and the observed embedding at high-surprisal
token positions — i.e., where the model's prediction diverged most from
what the student actually wrote.

The research question is whether these delta vectors cluster into
interpretable linguistic deltas (along the lines of "model predicted determiner; student provided noun" or "model predicted pronoun; student provided proper noun").

Pipeline overview:
    1. pilot.ipynb extracts 55,998 delta vectors (768-dim) and metadata
       from a stratified sample of 999 ELLIPSE essays
    2. This CLI handles downstream exploration:
       - `reduce`: UMAP dimensionality reduction (slow, run once)
       - `cluster`: HDBSCAN clustering (fast, iterate with different params)
       - `show`: KWIC concordance lines for inspecting individual clusters

Intended workflow:
    # Step 1: Run UMAP (once, ~5-10 min)
    python concordance.py reduce

    # Step 2: Try clustering with default params
    python concordance.py cluster
    python concordance.py show 0

    # Step 3: Too many clusters? Increase min_cluster_size
    python concordance.py cluster --min-cluster-size 500
    python concordance.py show 0
    python concordance.py show 1 --filter-pos NOUN --filter-prof low

    # Step 4: Try leaf method for more even-sized clusters
    python concordance.py cluster --min-cluster-size 300 --method leaf
    python concordance.py show 0 --sort surprisal --n 100

Success criteria: 3-5+ non-trivial clusters that can be labeled with
recognizable linguistic error categories, with cluster membership
distributions that differ across proficiency levels (low/mid/high).

Source data (read-only, from pilot.ipynb):
    data/pilot_deltas.npy        — (N, 768) float32 transformation vectors
    data/pilot_metadata.parquet  — per-token metadata (token, surprisal,
                                   POS, dep_rel, context, proficiency,
                                   top-3 predictions)

Cached artifacts (in data/umap_cache/):
    umap_2d.npy, umap_10d.npy, umap_params.json   — from `reduce`
    cluster_labels.npy, cluster_params.json         — from `cluster`

Environment:
    Requires umap-learn and hdbscan (plus numpy, pandas, scikit-learn).
    UMAP reduction is already cached in data/umap_cache/ — only `cluster`
    and `show` need to be re-run during iterative exploration.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.paths import DATA_DIR

CACHE_DIR = DATA_DIR / "umap_cache"
DELTAS_PATH = DATA_DIR / "pilot_deltas.npy"
METADATA_PATH = DATA_DIR / "pilot_metadata.parquet"


def load_metadata(path=METADATA_PATH):
    """Load the pilot metadata DataFrame."""
    return pd.read_parquet(path)


# ── reduce ──────────────────────────────────────────────────────────────

def cmd_reduce(args):
    """Run UMAP dimensionality reduction and cache results."""
    import umap

    print(f"Loading deltas from {DELTAS_PATH}")
    deltas = np.load(DELTAS_PATH)
    print(f"  shape: {deltas.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "seed": args.seed,
    }

    # 2D for visualization
    print(f"Running 2D UMAP (n_neighbors={args.n_neighbors}, "
          f"min_dist={args.min_dist}, metric={args.metric})...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    embedding_2d = reducer_2d.fit_transform(deltas)
    np.save(CACHE_DIR / "umap_2d.npy", embedding_2d.astype(np.float32))
    print(f"  saved umap_2d.npy {embedding_2d.shape}")

    # 10D for clustering
    print("Running 10D UMAP...")
    reducer_10d = umap.UMAP(
        n_components=10,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    embedding_10d = reducer_10d.fit_transform(deltas)
    np.save(CACHE_DIR / "umap_10d.npy", embedding_10d.astype(np.float32))
    print(f"  saved umap_10d.npy {embedding_10d.shape}")

    # Save params
    with open(CACHE_DIR / "umap_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print("  saved umap_params.json")

    print("Done.")


# ── cluster ─────────────────────────────────────────────────────────────

def cmd_cluster(args):
    """Run HDBSCAN on cached UMAP embeddings and print summary."""
    import hdbscan

    umap_path = CACHE_DIR / "umap_10d.npy"
    if not umap_path.exists():
        print(f"ERROR: {umap_path} not found.")
        print("Run `python concordance.py reduce` first.")
        sys.exit(1)

    print(f"Loading 10D UMAP embeddings from {umap_path}")
    embedding_10d = np.load(umap_path)
    print(f"  shape: {embedding_10d.shape}")

    print(f"Running HDBSCAN (min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples}, method={args.method})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.method,
    )
    labels = clusterer.fit_predict(embedding_10d)

    np.save(CACHE_DIR / "cluster_labels.npy", labels.astype(np.int32))
    with open(CACHE_DIR / "cluster_params.json", "w") as f:
        json.dump({
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "method": args.method,
        }, f, indent=2)

    # Summary stats
    meta = load_metadata()
    total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clustered = total - n_noise
    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    print(f"\n=== CLUSTER SUMMARY ===")
    print(f"total_vectors:  {total}")
    print(f"clustered:      {n_clustered} ({100 * n_clustered / total:.1f}%)")
    print(f"noise:          {n_noise} ({100 * n_noise / total:.1f}%)")
    print(f"n_clusters:     {n_clusters}")
    print()

    if n_clusters == 0:
        print("No clusters found. Try decreasing --min-cluster-size.")
        return

    # Per-cluster summary, sorted by size descending
    meta = meta.copy()
    meta["_label"] = labels
    cluster_sizes = [(cid, int((labels == cid).sum())) for cid in unique_labels]
    cluster_sizes.sort(key=lambda x: -x[1])

    if args.top is not None:
        cluster_sizes = cluster_sizes[:args.top]

    # Header
    print(f"{'CLUSTER':>7s}  {'SIZE':>5s}  {'SURP_MEAN':>9s}  {'SURP_SD':>7s}  "
          f"{'LOW%':>5s}  {'MID%':>5s}  {'HIGH%':>5s}  "
          f"{'TOP_POS':<16s}{'TOP_DEP':<16s}")

    for cid, size in cluster_sizes:
        cdf = meta[meta["_label"] == cid]
        surp_mean = cdf["surprisal"].mean()
        surp_sd = cdf["surprisal"].std()

        prof_counts = cdf["prof_tertile"].value_counts(normalize=True)
        low_pct = 100 * prof_counts.get("low", 0)
        mid_pct = 100 * prof_counts.get("mid", 0)
        high_pct = 100 * prof_counts.get("high", 0)

        pos_top = cdf["pos_tag"].value_counts()
        top_pos = f"{pos_top.index[0]}({100 * pos_top.iloc[0] / size:.0f}%)"

        dep_top = cdf["dep_rel"].value_counts()
        top_dep = f"{dep_top.index[0]}({100 * dep_top.iloc[0] / size:.0f}%)"

        print(f"{cid:>7d}  {size:>5d}  {surp_mean:>9.2f}  {surp_sd:>7.2f}  "
              f"{low_pct:>5.1f}  {mid_pct:>5.1f}  {high_pct:>5.1f}  "
              f"{top_pos:<16s}{top_dep:<16s}")

    if args.top is not None and args.top < n_clusters:
        print(f"\n(showing top {args.top} of {n_clusters} clusters)")


# ── show ────────────────────────────────────────────────────────────────

def print_cluster_stats(cdf):
    """Print POS, dep, and proficiency distributions for a cluster."""
    print(f"\n--- CLUSTER STATS ({len(cdf)} tokens) ---")
    print(f"Mean surprisal: {cdf['surprisal'].mean():.2f} "
          f"(SD {cdf['surprisal'].std():.2f})")

    pos_counts = cdf["pos_tag"].value_counts()
    pos_parts = [f"{tag}({100 * n / len(cdf):.0f}%)"
                 for tag, n in pos_counts.head(8).items()]
    print(f"POS:  {', '.join(pos_parts)}")

    dep_counts = cdf["dep_rel"].value_counts()
    dep_parts = [f"{dep}({100 * n / len(cdf):.0f}%)"
                 for dep, n in dep_counts.head(8).items()]
    print(f"DEP:  {', '.join(dep_parts)}")

    prof_counts = cdf["prof_tertile"].value_counts(normalize=True)
    prof_parts = [f"{t}: {100 * p:.1f}%" for t, p in prof_counts.items()]
    print(f"PROF: {', '.join(prof_parts)}")


def format_kwic_line(row, idx, context_chars=35, keyword_width=16):
    """Format a single KWIC concordance line."""
    # Left context: right-aligned, truncated with ... prefix
    left = row["left_context"] if isinstance(row["left_context"], str) else ""
    if len(left) > context_chars:
        left = "..." + left[-(context_chars - 3):]
    left = left.rjust(context_chars)

    # Keyword: centered in brackets
    token = row["token_text"]
    if len(token) > keyword_width - 4:
        token = token[:keyword_width - 4]
    keyword = f"[{token:^{keyword_width - 2}s}]"

    # Right context: left-aligned, truncated
    right = row["right_context"] if isinstance(row["right_context"], str) else ""
    if len(right) > context_chars:
        right = right[:context_chars - 3] + "..."
    right = right.ljust(context_chars)

    # Top-3 predictions
    top3_predicted = row["top3_predicted"]
    top3_probs = row["top3_probs"]
    preds = []
    for i in range(3):
        if i < len(top3_predicted):
            tok = top3_predicted[i].strip()
            prob = top3_probs[i]
            preds.append(f"{tok}({prob:.2f})")
        else:
            preds.append("")

    return (f"{idx:>3d}  {left}  {keyword}  {right}  "
            f"{row['surprisal']:>5.2f}  {row['pos_tag']:<5s} "
            f"{row['dep_rel']:<7s} {row['prof_tertile']:<5s} "
            f"{preds[0]:<16s}{preds[1]:<16s}{preds[2]:<16s}")


def cmd_show(args):
    """Display KWIC concordance lines for a cluster."""
    labels_path = CACHE_DIR / "cluster_labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.")
        print("Run `python concordance.py cluster` first.")
        sys.exit(1)

    labels = np.load(labels_path)
    meta = load_metadata()

    # Validate cluster ID
    unique_labels = set(labels)
    unique_labels.discard(-1)
    if args.cluster_id not in unique_labels:
        valid = sorted(unique_labels)
        preview = valid[:20]
        suffix = f"... ({len(valid)} total)" if len(valid) > 20 else ""
        print(f"ERROR: cluster {args.cluster_id} not found.")
        print(f"Valid cluster IDs: {preview}{suffix}")
        sys.exit(1)

    # Select cluster — use boolean mask for aligned indexing with deltas
    cmask = labels == args.cluster_id
    cdf = meta[cmask].reset_index(drop=True)

    # Print full cluster stats (before filtering)
    print_cluster_stats(cdf)

    # Apply filters (OR within same filter type, AND across)
    display_df = cdf
    filter_desc = []

    if args.filter_pos:
        display_df = display_df[display_df["pos_tag"].isin(args.filter_pos)]
        filter_desc.append(f"POS={','.join(args.filter_pos)}")

    if args.filter_dep:
        display_df = display_df[display_df["dep_rel"].isin(args.filter_dep)]
        filter_desc.append(f"DEP={','.join(args.filter_dep)}")

    if args.filter_prof:
        display_df = display_df[display_df["prof_tertile"].isin(args.filter_prof)]
        filter_desc.append(f"PROF={','.join(args.filter_prof)}")

    if len(display_df) == 0:
        print(f"\nNo tokens match filters: {', '.join(filter_desc)}")
        return

    # Sort
    if args.sort == "centroid":
        from sklearn.metrics import pairwise_distances
        deltas = np.load(DELTAS_PATH)
        cluster_deltas = deltas[cmask]
        centroid = cluster_deltas.mean(axis=0)
        # Compute distances for filtered subset using aligned indices
        filtered_deltas = cluster_deltas[display_df.index.values]
        dists = pairwise_distances(
            filtered_deltas, centroid.reshape(1, -1), metric="cosine"
        ).ravel()
        display_df = display_df.copy()
        display_df["_sort_dist"] = dists
        display_df = display_df.sort_values("_sort_dist")
    elif args.sort == "surprisal":
        display_df = display_df.sort_values("surprisal", ascending=False)
    elif args.sort == "random":
        display_df = display_df.sample(frac=1, random_state=42)

    total_matching = len(display_df)
    display_df = display_df.head(args.n)

    # Print concordance
    filter_str = f", filtered by {', '.join(filter_desc)}" if filter_desc else ""
    print(f"\n--- CONCORDANCE ({len(display_df)} of {total_matching}, "
          f"sorted by {args.sort}{filter_str}) ---")

    cc = args.context_chars
    kw = 16
    print(f"{'#':>3s}  {'LEFT_CONTEXT':>{cc}s}  {'KEYWORD':^{kw + 2}s}  "
          f"{'RIGHT_CONTEXT':<{cc}s}  {'SURP':>5s}  {'POS':<5s} "
          f"{'DEP':<7s} {'PROF':<5s} "
          f"{'PRED_1':<16s}{'PRED_2':<16s}{'PRED_3':<16s}")

    for i, (_, row) in enumerate(display_df.iterrows(), 1):
        print(format_kwic_line(row, i, context_chars=cc, keyword_width=kw))


# ── parser ──────────────────────────────────────────────────────────────

def build_parser():
    """Build the argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="CLI concordancer for vector transformation clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # reduce
    p_reduce = sub.add_parser("reduce", help="Run UMAP dimensionality reduction")
    p_reduce.add_argument("--n-neighbors", type=int, default=30)
    p_reduce.add_argument("--min-dist", type=float, default=0.0)
    p_reduce.add_argument("--metric", type=str, default="cosine")
    p_reduce.add_argument("--seed", type=int, default=42)

    # cluster
    p_cluster = sub.add_parser("cluster", help="Run HDBSCAN clustering")
    p_cluster.add_argument("--min-cluster-size", type=int, default=100)
    p_cluster.add_argument("--min-samples", type=int, default=10)
    p_cluster.add_argument("--method", choices=["eom", "leaf"], default="eom")
    p_cluster.add_argument("--top", type=int, default=None,
                           help="Show only top N largest clusters")

    # show
    p_show = sub.add_parser("show", help="Show KWIC concordance for a cluster")
    p_show.add_argument("cluster_id", type=int, help="Cluster ID to inspect")
    p_show.add_argument("--n", type=int, default=50,
                        help="Number of concordance lines")
    p_show.add_argument("--sort", choices=["centroid", "surprisal", "random"],
                        default="centroid")
    p_show.add_argument("--context-chars", type=int, default=35)
    p_show.add_argument("--filter-pos", action="append", default=None,
                        help="Filter by POS tag (repeatable)")
    p_show.add_argument("--filter-dep", action="append", default=None,
                        help="Filter by dependency relation (repeatable)")
    p_show.add_argument("--filter-prof", action="append", default=None,
                        help="Filter by proficiency tertile (repeatable)")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "reduce":
        cmd_reduce(args)
    elif args.command == "cluster":
        cmd_cluster(args)
    elif args.command == "show":
        cmd_show(args)
