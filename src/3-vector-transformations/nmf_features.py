"""NMF decomposition of prediction-error delta vectors.

Complementary to the UMAP + HDBSCAN pipeline in concordance.py. NMF gives
soft/additive feature assignments — a token can load on multiple components —
and operates directly on the 768-dim delta vectors without UMAP preprocessing.

Subcommands:
    fit      Run NMF decomposition on the delta vectors
    summary  Overview table of all components
    show     Inspect a single component (KWIC concordance + stats)

Workflow:
    PY=/home/jovyan/conda_envs/hf/bin/python

    # Fit NMF (once, ~2-3 min)
    $PY nmf_features.py fit --n-components 50

    # Browse components
    $PY nmf_features.py summary
    $PY nmf_features.py show 0 --n 50
    $PY nmf_features.py show 3 --filter-pos NOUN --n 30

Source data (read-only, from pilot.ipynb):
    data/pilot_deltas.npy        — (55998, 768) float32 delta vectors
    data/pilot_metadata.parquet  — per-token metadata

Cached artifacts (in data/nmf_cache/):
    W.npy          — token loadings (N × k), float32
    H.npy          — component directions (k × 1536), float32
    params.json    — hyperparameters + reconstruction error + sparsity
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

CACHE_DIR = DATA_DIR / "nmf_cache"
DELTAS_PATH = DATA_DIR / "pilot_deltas.npy"
METADATA_PATH = DATA_DIR / "pilot_metadata.parquet"


def load_metadata(path=METADATA_PATH):
    """Load the pilot metadata DataFrame."""
    return pd.read_parquet(path)


def posneg_split(deltas):
    """Expand (N, d) deltas to (N, 2d) via hstack(max(0, x), max(0, -x)).

    Preserves directional information while satisfying NMF non-negativity.
    """
    return np.hstack([np.maximum(0, deltas), np.maximum(0, -deltas)])


# ── fit ────────────────────────────────────────────────────────────────

def cmd_fit(args):
    """Run NMF decomposition and cache W, H matrices."""
    from sklearn.decomposition import NMF

    print(f"Loading deltas from {DELTAS_PATH}")
    deltas = np.load(DELTAS_PATH)
    print(f"  shape: {deltas.shape}")

    print("Applying positive/negative split...")
    X = posneg_split(deltas)
    print(f"  expanded shape: {X.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    k = args.n_components
    print(f"Fitting NMF (k={k}, max_iter={args.max_iter}, seed={args.seed})...")
    t0 = time.time()

    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=args.max_iter,
        tol=1e-4,
        random_state=args.seed,
    )
    W = model.fit_transform(X)
    H = model.components_

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  (n_iter={model.n_iter_})")

    # Reconstruction error
    recon_err = model.reconstruction_err_
    x_norm = np.linalg.norm(X, "fro")
    err_ratio = recon_err / x_norm

    # Sparsity: fraction of W entries below 1e-6
    w_sparsity = float((W < 1e-6).mean())

    print(f"  reconstruction error: {recon_err:.2f}")
    print(f"  error ratio (err / ||X||_F): {err_ratio:.4f}")
    print(f"  W sparsity (frac < 1e-6): {w_sparsity:.3f}")

    # Save
    np.save(CACHE_DIR / "W.npy", W.astype(np.float32))
    np.save(CACHE_DIR / "H.npy", H.astype(np.float32))
    print(f"  saved W.npy {W.shape}, H.npy {H.shape}")

    params = {
        "n_components": k,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "n_iter_actual": int(model.n_iter_),
        "reconstruction_err": float(recon_err),
        "err_ratio": float(err_ratio),
        "w_sparsity": float(w_sparsity),
        "elapsed_sec": round(elapsed, 1),
        "input_shape": list(X.shape),
    }
    with open(CACHE_DIR / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    print("  saved params.json")

    print("Done.")


# ── summary ────────────────────────────────────────────────────────────

def cmd_summary(args):
    """Print overview table for all NMF components."""
    from scipy.stats import spearmanr

    W = np.load(CACHE_DIR / "W.npy")
    meta = load_metadata()
    k = W.shape[1]
    top_n = args.top_n

    # Map proficiency to numeric for correlation
    prof_map = {"low": 1, "mid": 2, "high": 3}
    prof_numeric = meta["prof_tertile"].map(prof_map).values

    print(f"{'COMP':>4s}  {'TOP_LOAD':>8s}  {'SURP':>5s}  "
          f"{'LOW%':>5s}  {'MID%':>5s}  {'HIGH%':>5s}  {'RHO':>6s}  "
          f"{'TOP_POS':<16s}{'TOP_TOKEN':<16s}")

    for c in range(k):
        loadings = W[:, c]
        top_idx = np.argsort(loadings)[-top_n:][::-1]
        top_meta = meta.iloc[top_idx]

        top_load = float(loadings[top_idx].mean())
        surp = float(top_meta["surprisal"].mean())

        prof_counts = top_meta["prof_tertile"].value_counts(normalize=True)
        low_pct = 100 * prof_counts.get("low", 0)
        mid_pct = 100 * prof_counts.get("mid", 0)
        high_pct = 100 * prof_counts.get("high", 0)

        # Spearman correlation: component loading vs proficiency (all tokens)
        rho, _ = spearmanr(loadings, prof_numeric)

        # Top POS among top-N tokens
        pos_top = top_meta["pos_tag"].value_counts()
        top_pos_parts = [f"{tag}({100 * n / top_n:.0f}%)"
                         for tag, n in pos_top.head(3).items()]
        top_pos_str = ",".join(top_pos_parts)

        # Most common token text among top-N
        top_token = top_meta["token_text"].value_counts().index[0]

        print(f"{c:>4d}  {top_load:>8.3f}  {surp:>5.2f}  "
              f"{low_pct:>5.1f}  {mid_pct:>5.1f}  {high_pct:>5.1f}  "
              f"{rho:>6.3f}  {top_pos_str:<16s}{top_token:<16s}")


# ── show ───────────────────────────────────────────────────────────────

def print_component_stats(cdf, component_id, loading_col="_loading"):
    """Print POS, dep, and proficiency distributions for a component."""
    n = len(cdf)
    print(f"\n--- COMPONENT {component_id} STATS (top {n} tokens) ---")
    print(f"Mean loading: {cdf[loading_col].mean():.4f} "
          f"(max {cdf[loading_col].max():.4f})")
    print(f"Mean surprisal: {cdf['surprisal'].mean():.2f} "
          f"(SD {cdf['surprisal'].std():.2f})")

    pos_counts = cdf["pos_tag"].value_counts()
    pos_parts = [f"{tag}({100 * cnt / n:.0f}%)"
                 for tag, cnt in pos_counts.head(8).items()]
    print(f"POS:  {', '.join(pos_parts)}")

    dep_counts = cdf["dep_rel"].value_counts()
    dep_parts = [f"{dep}({100 * cnt / n:.0f}%)"
                 for dep, cnt in dep_counts.head(8).items()]
    print(f"DEP:  {', '.join(dep_parts)}")

    prof_counts = cdf["prof_tertile"].value_counts(normalize=True)
    prof_parts = [f"{t}: {100 * p:.1f}%" for t, p in prof_counts.items()]
    print(f"PROF: {', '.join(prof_parts)}")


def cmd_show(args):
    """Display KWIC concordance lines for a single NMF component."""
    from concordance import format_kwic_line

    w_path = CACHE_DIR / "W.npy"
    if not w_path.exists():
        print(f"ERROR: {w_path} not found.")
        print("Run `python nmf_features.py fit` first.")
        sys.exit(1)

    W = np.load(w_path)
    meta = load_metadata()
    k = W.shape[1]
    cid = args.component_id

    if cid < 0 or cid >= k:
        print(f"ERROR: component {cid} out of range [0, {k - 1}].")
        sys.exit(1)

    loadings = W[:, cid]
    top_n = args.n

    # Get top-N tokens by loading
    top_idx = np.argsort(loadings)[-top_n:][::-1]
    cdf = meta.iloc[top_idx].copy()
    cdf["_loading"] = loadings[top_idx]

    # Print stats before filtering
    print_component_stats(cdf, cid)

    # Apply filters
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

    total_matching = len(display_df)

    # Print concordance
    filter_str = f", filtered by {', '.join(filter_desc)}" if filter_desc else ""
    print(f"\n--- CONCORDANCE ({total_matching} tokens{filter_str}) ---")

    cc = args.context_chars
    kw = 16
    print(f"{'#':>3s}  {'LEFT_CONTEXT':>{cc}s}  {'KEYWORD':^{kw + 2}s}  "
          f"{'RIGHT_CONTEXT':<{cc}s}  {'SURP':>5s}  {'POS':<5s} "
          f"{'DEP':<7s} {'PROF':<5s} "
          f"{'PRED_1':<16s}{'PRED_2':<16s}{'PRED_3':<16s}")

    for i, (_, row) in enumerate(display_df.iterrows(), 1):
        print(format_kwic_line(row, i, context_chars=cc, keyword_width=kw))


# ── parser ─────────────────────────────────────────────────────────────

def build_parser():
    """Build the argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="NMF decomposition of prediction-error delta vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fit
    p_fit = sub.add_parser("fit", help="Run NMF decomposition")
    p_fit.add_argument("--n-components", type=int, default=50,
                       help="Number of NMF components (default: 50)")
    p_fit.add_argument("--max-iter", type=int, default=500,
                       help="Maximum NMF iterations (default: 500)")
    p_fit.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    # summary
    p_summary = sub.add_parser("summary", help="Overview of all components")
    p_summary.add_argument("--top-n", type=int, default=50,
                           help="Tokens per component for stats (default: 50)")

    # show
    p_show = sub.add_parser("show", help="Inspect a single component")
    p_show.add_argument("component_id", type=int,
                        help="Component index to inspect")
    p_show.add_argument("--n", type=int, default=50,
                        help="Number of top tokens to display")
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

    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "show":
        cmd_show(args)
