"""Step 2 - turn the respondent x item matrix into an opinion network.

Design decisions, in the order they are applied
-----------------------------------------------
1. Naive similarity (1 - normalised L1 distance) is computed first only to
   *demonstrate* that it is degenerate: the survey has a heavy agreement bias,
   so on the raw scale almost every pair of students looks alike.
2. Double centring removes the two nuisance effects that cause that:
      column z-scoring  -> removes item popularity ("everyone agrees with V15")
      row centring      -> removes acquiescence style (some students tick
                           "Strongly Agree" for everything)
   What survives is each student's *relative* position on each statement.
3. Edge weight = Pearson correlation between two double-centred profiles.
4. The retention threshold is not chosen by eye. A permutation null model
   destroys inter-respondent structure while preserving each item's marginal
   distribution; an edge is kept only if its correlation exceeds the
   NULL_PERCENTILE of the null correlation distribution.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd

import config as C

RNG = np.random.default_rng(C.RANDOM_SEED)


# ------------------------------------------------------------------ helpers
def load_matrix() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED / "responses_numeric.csv", index_col="respondent")


def naive_similarity(X: np.ndarray) -> np.ndarray:
    """1 - mean |difference| / 4.  The obvious baseline, shown to be degenerate."""
    n = X.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        S[i] = 1.0 - np.abs(X[i] - X).mean(axis=1) / 4.0
    return S


def double_centre(X: np.ndarray) -> np.ndarray:
    """z-score each item, then remove each respondent's mean response style."""
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=0) == 0, 1.0, X.std(axis=0, ddof=0))
    return Z - Z.mean(axis=1, keepdims=True)


def correlation(Z: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows, with the diagonal zeroed."""
    Zc = Z - Z.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Zc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    R = (Zc / norm) @ (Zc / norm).T
    np.fill_diagonal(R, 0.0)
    return R


def upper(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


# ----------------------------------------------------------- null model
def null_distribution(X: np.ndarray, n_perm: int = C.NULL_PERMUTATIONS) -> np.ndarray:
    """Correlations obtained when each item is independently shuffled.

    Shuffling within a column keeps that item's marginal distribution (and so
    its popularity and its variance) but destroys any association *between*
    respondents. Correlations from this ensemble are what the pipeline would
    produce from structureless data.
    """
    pool = []
    for _ in range(n_perm):
        Xp = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            Xp[:, j] = RNG.permutation(X[:, j])
        pool.append(upper(correlation(double_centre(Xp))))
    return np.concatenate(pool)


def empirical_p_values(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided p = P(null >= observed), estimated from the sorted null pool."""
    srt = np.sort(null)
    ranks = np.searchsorted(srt, obs, side="left")
    return (len(srt) - ranks + 1) / (len(srt) + 1)


def benjamini_hochberg(p: np.ndarray, alpha: float) -> np.ndarray:
    """Return a boolean mask of hypotheses rejected at FDR level alpha."""
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        keep[order[: cutoff + 1]] = True
    return keep


# ----------------------------------------------------------- graph building
def graph_from_matrix(R: np.ndarray, labels, threshold: float) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(labels)
    iu = np.triu_indices_from(R, k=1)
    for i, j, w in zip(iu[0], iu[1], R[iu]):
        if w > threshold:
            G.add_edge(labels[i], labels[j], weight=float(w), distance=float(1.0 - w))
    return G


def sweep(R: np.ndarray, labels) -> pd.DataFrame:
    """How the network responds to the threshold - reported for transparency."""
    rows = []
    for t in C.THRESHOLD_SWEEP:
        G = graph_from_matrix(R, labels, t)
        comps = list(nx.connected_components(G))
        giant = max((len(c) for c in comps), default=0)
        rows.append(
            {
                "threshold": t,
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "isolates": sum(1 for n in G if G.degree(n) == 0),
                "components": len(comps),
                "giant_component_frac": giant / G.number_of_nodes(),
                "avg_clustering": nx.average_clustering(G, weight=None),
            }
        )
    return pd.DataFrame(rows)


def main() -> dict:
    df = load_matrix()
    labels = df.index.astype(str).tolist()
    X = df.to_numpy(dtype=float)

    # 1 - the degenerate baseline ------------------------------------------
    S_naive = naive_similarity(X)
    naive_vals = upper(S_naive)

    # 2/3 - the similarity actually used -----------------------------------
    Z = double_centre(X)
    R = correlation(Z)
    obs = upper(R)

    # 4 - calibrate the threshold against the null -------------------------
    null = null_distribution(X)
    tau = float(np.percentile(null, C.NULL_PERCENTILE))
    p = empirical_p_values(obs, null)
    fdr_mask = benjamini_hochberg(p, C.FDR_ALPHA)
    tau_fdr = float(obs[fdr_mask].min()) if fdr_mask.any() else float("inf")

    G = graph_from_matrix(R, labels, tau)
    G.graph.update(
        threshold=tau, null_percentile=C.NULL_PERCENTILE, n_permutations=C.NULL_PERMUTATIONS
    )

    # per-topic networks, each calibrated against its own null --------------
    topic_graphs, topic_meta = {}, {}
    for letter in C.TOPICS:
        cols = [c for c in df.columns if c.startswith(letter)]
        Xt = df[cols].to_numpy(dtype=float)
        Rt = correlation(double_centre(Xt))
        nt = null_distribution(Xt, n_perm=200)
        tt = float(np.percentile(nt, C.NULL_PERCENTILE))
        Gt = graph_from_matrix(Rt, labels, tt)
        topic_graphs[letter] = Gt
        topic_meta[letter] = {
            "threshold": tt,
            "edges": Gt.number_of_edges(),
            "density": nx.density(Gt),
            "mean_weight": float(np.mean([d["weight"] for *_, d in Gt.edges(data=True)]))
            if Gt.number_of_edges()
            else 0.0,
        }
        nx.write_graphml(Gt, C.NETWORK / f"opinion_network_{letter}.graphml")

    # item-item network: which statements move together ---------------------
    # Calibrated exactly like the respondent network: same one-sided rule, same
    # percentile, an ensemble of permutations rather than a single draw.
    Ritem = correlation(double_centre(X).T)
    item_null_pool = np.concatenate(
        [
            upper(correlation(double_centre(RNG.permuted(X, axis=0)).T))
            for _ in range(200)
        ]
    )
    item_tau = float(np.percentile(item_null_pool, C.NULL_PERCENTILE))
    Gitem = graph_from_matrix(Ritem, list(df.columns), item_tau)

    # ------------------------------------------------------------- persist
    nx.write_graphml(G, C.NETWORK / "opinion_network_main.graphml")
    nx.write_graphml(Gitem, C.NETWORK / "item_network.graphml")
    np.save(C.NETWORK / "similarity_matrix.npy", R)
    # The full null pool is ~1.9M values (15 MB); only its shape is ever needed
    # again, so a large random subsample is persisted instead.
    np.save(C.NETWORK / "null_pool_sample.npy",
            RNG.choice(null, size=min(200_000, null.size), replace=False))
    np.save(C.NETWORK / "naive_similarity.npy", S_naive)
    np.save(C.NETWORK / "item_similarity.npy", Ritem)
    pd.DataFrame(R, index=labels, columns=labels).to_csv(C.TABLES / "similarity_matrix.csv")

    sweep_df = sweep(R, labels)
    sweep_df.to_csv(C.TABLES / "threshold_sweep.csv", index=False)

    nx.to_pandas_edgelist(G).to_csv(C.NETWORK / "edges_main.csv", index=False)

    comps = list(nx.connected_components(G))
    meta = {
        "n_nodes": G.number_of_nodes(),
        "n_possible_pairs": int(len(labels) * (len(labels) - 1) / 2),
        "naive_similarity_mean": float(naive_vals.mean()),
        "naive_similarity_min": float(naive_vals.min()),
        "naive_frac_above_0_7": float((naive_vals > 0.70).mean()),
        "corr_mean": float(obs.mean()),
        "corr_std": float(obs.std(ddof=1)),
        "corr_min": float(obs.min()),
        "corr_max": float(obs.max()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "threshold_tau": tau,
        "threshold_fdr": tau_fdr,
        "n_significant_fdr": int(fdr_mask.sum()),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "isolates": sum(1 for n in G if G.degree(n) == 0),
        "n_components": len(comps),
        "giant_component": max(len(c) for c in comps),
        "mean_edge_weight": float(np.mean([d["weight"] for *_, d in G.edges(data=True)])),
        "topics": topic_meta,
        "item_network": {
            "threshold": item_tau,
            "nodes": Gitem.number_of_nodes(),
            "edges": Gitem.number_of_edges(),
        },
    }
    (C.NETWORK / "build_report.json").write_text(json.dumps(meta, indent=2))

    print(f"[build] naive similarity       : mean {meta['naive_similarity_mean']:.3f}, "
          f"min {meta['naive_similarity_min']:.3f}, "
          f"{meta['naive_frac_above_0_7']:.0%} of pairs above 0.70  <- degenerate")
    print(f"[build] corrected correlation  : mean {meta['corr_mean']:+.3f} "
          f"(sd {meta['corr_std']:.3f}), range [{meta['corr_min']:+.3f}, {meta['corr_max']:+.3f}]")
    print(f"[build] null model             : mean {meta['null_mean']:+.4f} "
          f"(sd {meta['null_std']:.3f}) from {C.NULL_PERMUTATIONS} permutations")
    print(f"[build] threshold tau (p95)    : {tau:.4f}   | BH-FDR cut {tau_fdr:.4f} "
          f"({meta['n_significant_fdr']} significant pairs)")
    print(f"[build] main network           : {meta['n_nodes']} nodes, {meta['n_edges']} edges, "
          f"density {meta['density']:.3f}, {meta['isolates']} isolates, "
          f"giant component {meta['giant_component']}")
    for k, v in topic_meta.items():
        print(f"[build]   {k} ({C.TOPICS[k]:<17}): tau {v['threshold']:.3f}, "
              f"{v['edges']:4d} edges, density {v['density']:.3f}")
    return meta


if __name__ == "__main__":
    main()
